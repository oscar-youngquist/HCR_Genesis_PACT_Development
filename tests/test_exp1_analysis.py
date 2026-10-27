"""Synthetic numeric recordings and CPU-only notebook execution; no robot imports."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp1_analysis", ROOT / "notebooks/exp1_analysis.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def metadata(task="b1z1_pact", seed=0, condition="combined-disturbance", episodes=2):
    return dict(task=task, evaluation_seed=seed, checkpoint_training_seed=123,
                checkpoint=f"/{task}/model_100.pt", checkpoint_sha256=task + "-hash",
                backend="synthetic", dt=1., physics_dt=.5, prescribed_steps=4,
                evaluator_sha256="fixture-evaluator", utilities_sha256="fixture-utils",
                weights_unchanged=True, failure_reason_bits={"height": 2},
                conventions=dict(quaternion="xyzw", world="z-up", joint="configured order, rad and rad/s"),
                resolved_environment=dict(seed=seed, env=dict(num_envs=2), control=dict(decimation=2)),
                arguments=dict(condition=condition, load_range="in", episodes=episodes),
                evaluation=dict(ee_tolerance_m=.1, velocity_tolerance_m_s=.1, yaw_tolerance_rad_s=.2,
                                recovery_window_seconds=2., recovery_dwell_seconds=1., chunk_rows=2))


def trial_data(errors, failure=False, trial_id=0, condition="combined-disturbance", seed=0):
    n = len(errors)
    d = dict(trial_id=np.full(n, trial_id), step=np.arange(n), successor_time_seconds=np.arange(1, n+1, dtype=float),
             dt=np.ones(n), alive_for_metrics=np.ones(n, bool), valid=np.ones(n, bool),
             physical_failure=np.zeros(n, bool), artificial_reset=np.zeros(n, bool),
             ee_position_world=np.column_stack((errors, np.zeros((n, 2)))), ee_reference_world=np.zeros((n, 3)),
             base_velocity_body=np.zeros((n, 3)), base_angular_velocity_body=np.zeros((n, 3)), command_nominal=np.zeros((n, 3)),
             base_quaternion_xyzw=np.tile([0., 0., 0., 1.], (n, 1)), actual_clipping_limits=np.full((n, 2), 11.),
             joint_position=np.zeros((n, 2)), saturation_substep_count=np.tile([1, 0], (n, 1)),
             position_violation_substep_count=np.zeros((n, 2), int), velocity_violation_substep_count=np.zeros((n, 2), int),
             base_angular_velocity_world=np.column_stack((np.arange(n, dtype=float), np.zeros((n, 2)))),
             ee_velocity_world=np.column_stack((np.arange(n, dtype=float)*2, np.zeros((n, 2)))))
    if failure:
        d["physical_failure"][-1] = True
        d["alive_for_metrics"][-1] = False
    events = [dict(time=float(i), changed=condition != "nominal", load=[1.] * 9 if i < 3 else [0.] * 9) for i in (1, 2, 3)]
    t = dict(trial_id=trial_id, seed=seed, condition=condition, load_range="in", motion_mode="locomotion",
             physical_failure=failure, events=events, scenario_hash=f"{seed}-{trial_id}-{condition}")
    return d, t


def config(root, out, task="b1z1_pact"):
    return dict(input_roots=[str(root)], task=task, display_name=task, expected_seeds=[0, 1, 2],
                checkpoint_sha256=None, output_dir=str(out), histogram_edges=[-.1, 0., .1],
                histogram_frame="yaw", include_combined=True, acceleration_metrics=True)


def make_fixture(root, task="b1z1_pact"):
    for seed in range(3):
        for condition in a.CONDITIONS:
            directory = root / task / f"arbitrary-{seed}-{condition}"
            directory.mkdir(parents=True)
            meta = metadata(task, seed, condition)
            cases = [([.02]*4, False), ([.04]*4, False)] if seed == 0 else (
                [([.06, 9.], True), ([9.], True)] if seed == 1 else [([.1]*4, False), ([.2]*4, False)])
            arrays, trials = [], []
            for trial_id, (errors, failed) in enumerate(cases):
                d, t = trial_data(errors, failed, trial_id, condition, seed)
                result, *_ = a.trial_metrics(d, t, meta, True)
                t.update({k: result[k] for k in a.MAIN[:-1]})
                arrays.append(d)
                trials.append(t)
            data = {k: np.concatenate([d[k] for d in arrays]) for k in arrays[0]}
            # Chunk boundaries cut through episodes, and rows are out of temporal order.
            index = np.arange(len(data["step"]))[::-1]
            data = {k: v[index] for k, v in data.items()}
            np.savez_compressed(directory / "transitions-000000.npz", **data)
            a.write_json(directory / "records.json", dict(schema_version=1,
                fields={k: dict(shape=list(v.shape[1:]), dtype=str(v.dtype)) for k, v in data.items()},
                chunks=[dict(file="transitions-000000.npz", rows=len(index))]))
            a.write_json(directory / "metadata.json", meta)
            a.write_json(directory / "trials.json", trials)
            a.write_json(directory / "seed_summary.json", {f"{seed}/{condition}/in/locomotion": {
                **{k: a.mean([t[k] for t in trials]) for k in a.MAIN[:-1]},
                "physical_failure_rate": float(np.mean([t["physical_failure"] for t in trials]))}})
            (directory / "COMPLETE").write_text("complete")


@pytest.mark.parametrize("errors,failed,expected_error,score", [
    ([.05]*4, False, .05, 1.), ([.05, 9.], True, .05, .25),
    ([.05, .05, .05, 9.], True, .05, .75), ([9.], True, None, 0.)])
def test_failure_and_horizon(errors, failed, expected_error, score):
    d, t = trial_data(errors, failed)
    result, *_ = a.trial_metrics(d, t, metadata())
    assert result["ee_error_m"] == expected_error or result["ee_error_m"] == pytest.approx(expected_error)
    assert result["S_track"] == score and result["failure_rate"] == float(failed)
    if failed:
        assert all(not e["recovered"] and e["recovery_time_s"] is None for e in result["events"])
        assert result["events"][-1]["peak_m"] is None


def test_reset_safe_differences_and_yaw():
    v = np.array([[0., 0., 0.], [2., 0., 0.], [100., 0., 0.], [104., 0., 0.]])
    rms, n = a.acceleration_rms(v, np.array([0., 1., 4., 5.]), np.array([0, 1, 4, 5]), np.ones(4, bool), 1.)
    assert rms == pytest.approx(np.sqrt(10)) and n == 2
    rms, n = a.acceleration_rms(v, np.arange(4.), np.arange(4), np.array([True, True, False, True]), 1.)
    assert rms == 2. and n == 1
    np.testing.assert_allclose(a.yaw_frame(np.array([[1., 0., 0.]]), np.array([[0., 0., np.sqrt(.5), np.sqrt(.5)]])), [[0., -1., 0.]], atol=1e-15)


def test_histogram_tails_and_sample_sd():
    rows = a.histogram_rows(np.array([[-2., 0., 0.], [0., 0., 0.], [2., 0., 0.]]), np.array([-1., 0., 1.]), 0, "nominal", "in", "stationary")
    x = [r for r in rows if r["axis"] == "x"]
    assert sum(r["count"] for r in x) == 1 and x[0]["underflow"] == x[0]["overflow"] == 1
    assert sum(r["probability"] for r in x) == pytest.approx(1/3)
    assert a.seed_statistics([1.])["sd"] is None
    assert a.seed_statistics([0., 2.])["sd"] == pytest.approx(np.sqrt(2))


def test_ecdf_tails_and_equal_seed_boxes():
    rows = a.histogram_rows(np.array([[-2., 0., 0.], [0., 0., 0.], [2., 0., 0.]]),
                            np.array([-1., 0., 1.]), 0, "nominal", "in", "stationary")
    rows += a.histogram_rows(np.array([[.5, 0., 0.]]), np.array([-1., 0., 1.]),
                             1, "nominal", "in", "stationary")
    h = a.pd.DataFrame(rows)
    edges, cdf, sd, n = a.binned_ecdf(h[h.axis == "x"])
    np.testing.assert_allclose(cdf, [1/6, 1/6, 5/6])
    assert np.all(np.diff(cdf) >= 0) and n == 2 and sd is not None
    boxes = a.equal_seed_box_stats([np.zeros((100, 3)), np.full((1, 3), 10.)])
    assert boxes[0]["med"] == 0. and boxes[0]["q3"] == 10.  # Not dominated by the 100-sample seed.
    assert boxes[0]["whishi"] == 10. and boxes[0]["valid_samples"] == 101
    assert a.equal_seed_box_stats([np.empty((0, 3))]) == []


def test_unrecovered_event_and_complete_windows():
    d, t = trial_data([.3]*4)
    r, *_ = a.trial_metrics(d, t, metadata())
    assert all(not e["recovered"] and e["recovery_time_s"] is None for e in r["events"])
    assert r["events"][0]["complete_window"]
    assert r["events"][0]["peak_m"] == .3
    assert r["events"][0]["iae_m_s"] == .6
    # Gaps break dwell, even when both retained samples satisfy tolerances.
    d, t = trial_data([.01]*4)
    d = {k: v[[0, 2, 3]] for k, v in d.items()}
    m = metadata()
    m["evaluation"].update(recovery_window_seconds=3., recovery_dwell_seconds=2.)
    r, *_ = a.trial_metrics(d, t, m)
    assert not r["events"][0]["recovered"] and not r["events"][0]["complete_window"]


def test_duplicate_run_rejected(tmp_path):
    import shutil
    make_fixture(tmp_path / "raw")
    source = next((tmp_path / "raw").rglob("metadata.json")).parent
    shutil.copytree(source, tmp_path / "raw/duplicate")
    with pytest.raises(ValueError, match="Duplicate seed"):
        a.summarize(config(tmp_path / "raw", tmp_path / "out"))


def test_comparison_rejects_incompatible_seeds_and_units(tmp_path):
    for task in ("pact", "pos"):
        make_fixture(tmp_path / "raw", task)
        a.summarize(config(tmp_path / "raw", tmp_path / "summaries" / task, task))
    c = dict(summary_roots=[str(tmp_path / "summaries")], output_dir=str(tmp_path / "comparison"),
             method_order=["pact", "pos"], focused_methods=["pact", "pos"], load_range="in", motion="locomotion")
    path = tmp_path / "summaries/pos/summary.json"
    original = a.read_json(path)
    changed = copy.deepcopy(original)
    changed["per_seed"] = [r for r in changed["per_seed"] if r["seed"] != 2]
    a.write_json(path, changed)
    with pytest.raises(ValueError, match="seed coverage"):
        a.compare(c)
    changed = copy.deepcopy(original)
    changed["units"]["ee_error_m"] = "cm"
    a.write_json(path, changed)
    with pytest.raises(ValueError, match="Incompatible"):
        a.compare(c)


def test_summary_seed_weighting_and_limits(tmp_path):
    make_fixture(tmp_path / "raw")
    s = a.summarize(config(tmp_path / "raw", tmp_path / "out"))
    def metric(name):
        return next(r for r in s["aggregates"] if r["condition"] == "nominal" and r["motion"] == "locomotion" and r["metric"] == name)
    assert metric("ee_error_m")["mean"] == pytest.approx(.08)
    assert metric("ee_error_m")["sd"] == pytest.approx(np.std([.03, .06, .15], ddof=1))
    assert metric("failure_rate")["mean"] == pytest.approx(1/3)
    assert metric("actuator_saturation_pct")["mean"] == 25.
    assert all("valid=0" in message for message in s["diagnostics"])
    assert (tmp_path / "out/COMPLETE").exists()
    assert "NaN" not in (tmp_path / "out/summary.json").read_text()
    # Within-seed physical limits pool denominator counts, not trial frequencies.
    for p in (tmp_path / "raw").rglob("transitions-*.npz"):
        if "arbitrary-0-nominal" not in str(p):
            continue
        with np.load(p) as f:
            d = {k: f[k] for k in f.files}
        mask = d["trial_id"] == 1
        d["alive_for_metrics"][mask & (d["step"] > 0)] = False
        d["saturation_substep_count"][mask] = 2
        np.savez_compressed(p, **d)
    s = a.summarize(config(tmp_path / "raw", tmp_path / "out"))
    row = next(r for r in s["per_seed"] if r["seed"] == 0 and r["condition"] == "nominal" and r["motion"] == "locomotion")
    assert row["values"]["actuator_saturation_pct"] == 40.  # (4 + 4) / (16 + 4), not (25% + 100%)/2.


def test_protocol_and_duplicates_rejected(tmp_path):
    make_fixture(tmp_path / "raw")
    c = config(tmp_path / "raw", tmp_path / "out")
    path = next((tmp_path / "raw").rglob("metadata.json"))
    meta = a.read_json(path)
    meta["evaluation"]["ee_tolerance_m"] = .2
    a.write_json(path, meta)
    with pytest.raises(ValueError, match="protocols"):
        a.summarize(c)
    assert not (tmp_path / "out/COMPLETE").exists()


def test_notebooks_end_to_end(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    tasks = ["b1z1_pact", "b1z1_unifp_original", "b1z1_ppo_pos"]
    for task in tasks:
        make_fixture(tmp_path / "raw", task)
        settings = config(tmp_path / "raw", tmp_path / "summaries" / task, task)
        path = tmp_path / "analysis_config.json"
        a.write_json(path, settings)
        monkeypatch.setenv("EXP1_ANALYSIS_CONFIG", str(path))
        notebook = nbformat.read(ROOT / "notebooks/exp1_summarize_approach.ipynb", as_version=4)
        notebook.cells.append(nbformat.v4.new_code_cell("assert 'torch' not in sys.modules and 'isaaclab' not in sys.modules"))
        nbclient.NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=str(ROOT))
        nbformat.write(notebook, tmp_path / f"executed_{task}.ipynb")
        s = a.read_json(tmp_path / "summaries" / task / "summary.json")
        row = next(r for r in s["aggregates"] if r["condition"] == "nominal" and r["motion"] == "locomotion" and r["metric"] == "ee_error_m")
        assert row["mean"] == pytest.approx(.08)
    # No raw files remain accessible: comparison must use portable exports only.
    (tmp_path / "raw").rename(tmp_path / "hidden_raw")
    compare = dict(summary_roots=[str(tmp_path / "summaries")], output_dir=str(tmp_path / "comparison"),
                   method_order=tasks, focused_methods=tasks, load_range="in", motion="locomotion", response_table=True)
    path = tmp_path / "compare_config.json"
    a.write_json(path, compare)
    monkeypatch.setenv("EXP1_COMPARE_CONFIG", str(path))
    notebook = nbformat.read(ROOT / "notebooks/exp1_compare_approaches.ipynb", as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell(
        "assert all(ax.patches for ax in figure.axes)\n"
        "assert figure.axes[0].get_ylabel() == 'Samples per bin [%]'\n"
        "assert np.isclose(figure.axes[0].patches[1].get_height(), 100 * 5 / 6)"))
    nbclient.NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=str(ROOT))
    nbformat.write(notebook, tmp_path / "executed_comparison.ipynb")
    for name in ("main_table.csv", "main_table.tex", "physical_limits.tex", "disturbance_response.csv", "ee_distributions.pdf", "ee_distributions.png",
                 "ee_step.pdf", "ee_step.png", "ee_ecdf.pdf", "ee_ecdf.png", "ee_box.pdf", "ee_box.png", "ee_ecdf_data.csv", "ee_box_data.csv"):
        assert (tmp_path / "comparison" / name).stat().st_size > 0
    comparison = a.read_json(tmp_path / "comparison/comparison.json")
    assert comparison["seed_coverage"]["nominal"] == [0, 1, 2]
    plotted = a.pd.read_csv(tmp_path / "comparison/ee_distribution_data.csv")
    row = plotted[(plotted.task == "b1z1_pact") & (plotted.condition == "nominal") &
                  (plotted.axis == "x") & (plotted.bin_index == 1)].iloc[0]
    assert row.probability_mean == pytest.approx(5/6)  # Mean of seed masses 1, 1, 0.5, not pooled rows.
    assert row.outside_fraction == pytest.approx(1/6)
