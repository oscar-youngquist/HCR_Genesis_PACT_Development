"""Small CPU fixtures exercise portable Exp2 summaries and both notebooks."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import exp2_analysis as a
from test_exp1_analysis import metadata, trial_data

STAGES = ("stationary_unloaded", "start_ramp", "moving_unloaded", "load25_ee_motion",
          "load45", "load65_moving", "loaded_stop_ramp", "loaded_stationary", "released")


def fixture(root, task="b1z1_pact", seeds=(0, 1, 2), loads=("loaded", "unloaded")):
    for seed in seeds:
        for profile in a.PROFILES:
            for loading in loads:
                directory = root / task / f"{seed}-{profile}-{loading}"
                directory.mkdir(parents=True)
                meta = metadata(task, seed)
                meta.update(experiment=2, prescribed_steps=28, task_stages=dict(enumerate(STAGES)))
                meta["arguments"] = dict(profile=profile, loading=loading)
                meta["evaluation"].update(move_start_s=3., velocity_ramp_s=.5, load_times_s=[7., 11., 15., 23.],
                                          stop_s=19., finish_s=28., velocity_tolerance_m_s=.2, yaw_tolerance_rad_s=.3)
                arrays, trials = [], []
                # Include a terminal failure before/after events and unequal lengths.
                lengths = (28, 8) if seed == 0 else (28, 17) if seed == 1 else (28, 1)
                for tid, length in enumerate(lengths):
                    d, t = trial_data([.02*(seed+1)]*length, tid == 1, tid, seed=seed)
                    d["time_seconds"] = np.arange(length, dtype=float)
                    d["horizon_complete"] = (d["step"] == 27)
                    d["task_stage_successor"] = a.stage_ids(d["successor_time_seconds"], meta["evaluation"])
                    magnitude = np.select([d["time_seconds"] < 7, d["time_seconds"] < 11,
                                           d["time_seconds"] < 15, d["time_seconds"] < 23], [0., 25., 45., 65.], default=0.)
                    if loading == "unloaded":
                        magnitude *= 0
                    force = np.zeros((length, 3))
                    # Initial heading pi/2: task x maps to world y, not world x.
                    force[:, 2 if profile == "carrying" else 1] = magnitude * (1 if profile == "pulling" else -1)
                    d["ee_force_applied_world_interval_mean"] = force
                    d["command_nominal"][:, 0] = .4
                    d["base_velocity_body"][:, 0] = .4
                    t.update(motion_mode="loco-manipulation", profile=profile, loading=loading,
                             initial_task_heading_rad=np.pi/2, pair_id=f"{seed}-{tid}-{profile}")
                    t["events"] = [dict(time=time, changed=loading == "loaded", load=[value]+[0.]*8)
                                   for time, value in zip([7., 11., 15., 23.], [25., 45., 65., 0.])]
                    arrays.append(d)
                    trials.append(t)
                data = {k: np.concatenate([d[k] for d in arrays]) for k in arrays[0]}
                index = np.argsort(data["step"], kind="stable")
                data = {k: v[index] for k, v in data.items()}
                chunks = []
                for i, start in enumerate(range(0, len(index), 5)):
                    part = {k: v[start:start+5] for k, v in data.items()}
                    name = f"transitions-{i:06d}.npz"
                    np.savez_compressed(directory / name, **part)
                    chunks.append(dict(file=name, rows=len(part["step"])))
                a.shared.write_json(directory / "records.json", dict(fields={k: dict(shape=list(v.shape[1:]), dtype=str(v.dtype)) for k,v in data.items()}, chunks=chunks))
                a.shared.write_json(directory / "metadata.json", meta)
                a.shared.write_json(directory / "trials.json", trials)
                a.shared.write_json(directory / "seed_summary.json", {})
                (directory / "COMPLETE").write_text("ok")


def config(tmp, task="b1z1_pact"):
    return dict(input_roots=[str(tmp/"raw")], task=task, display_name={"b1z1_pact":"PILOT"}.get(task, task),
                expected_seeds=[0, 1, 2], loading=["loaded", "unloaded"], output_dir=str(tmp/"summaries"/task),
                histogram_edges=[-.01, 0., .01], acceleration_metrics=True)


def test_summary_profiles_failures_counts_and_alignment(tmp_path):
    fixture(tmp_path/"raw")
    s = a.summarize(config(tmp_path))
    assert len(s["per_seed"]) == 18
    assert s["provenance"][0]["training_seed"] == 123
    assert {r["loading"] for r in s["aggregates"]} == {"loaded", "unloaded"}
    item = next(r for r in s["aggregates"] if r["profile"] == "pushing" and r["loading"] == "loaded" and r["metric"] == "ee_error_m")
    assert item["mean"] == pytest.approx(.04) and item["sd"] == pytest.approx(.02)
    failed = next(t for t in s["trials"] if t["seed"] == 0 and t["trial_id"] == 1)
    assert failed["S_track"] == 7/28
    assert not any(e["recovered"] for e in failed["events"])
    assert not failed["events"][0]["complete_window"]
    assert failed["stages"]["loaded_stationary"]["S_track"] == 0
    assert failed["stages"]["loaded_stationary"]["ee_error_m"] is None
    out = Path(config(tmp_path)["output_dir"])
    curves = pd.read_csv(out/"time_series.csv")
    force = curves[(curves.seed == 0) & (curves.profile == "pushing") & (curves.loading == "loaded") & (curves.curve == "force")]
    assert force.loc[force.step == 7, "mean"].item() == -25
    assert force.loc[force.step == 7, "time_seconds"].item() == 7
    assert force.loc[force.step == 7, "count"].item() == 1
    ee = curves[(curves.seed == 0) & (curves.profile == "pushing") & (curves.loading == "loaded") & (curves.curve == "ee_error")]
    assert ee.loc[ee.step == 7, "time_seconds"].item() == 8
    assert (curves[(curves.loading == "unloaded") & (curves.curve == "force")]["sum"] == 0).all()
    hist = pd.read_csv(out/"ee_histograms.csv")
    x = hist[hist.axis == "x"]
    assert (x.overflow == x.valid_samples).all()  # Tails not silently renormalized away.
    assert "NaN" not in (out/"summary.json").read_text()


def test_pooled_limits_and_equal_seed_curves():
    rows = []
    for count, denominator in [(10, 100), (10, 10)]:
        rows.append(dict(ee_error_m=.1, valid_steps=1, events=[], stages={},
                         limits={m:dict(count=count, denominator=denominator) for m in a.shared.LIMITS}))
    metrics, counts, _ = a.reduce_trials(rows)
    assert metrics["actuator_saturation_pct"] == pytest.approx(100*20/110)
    assert counts["limit_totals"]["actuator_saturation_pct"]["denominator"] == 110
    for r in rows:
        r["limits"] = {m:dict(count=0, denominator=0) for m in a.shared.LIMITS}
    assert a.reduce_trials(rows)[0]["actuator_saturation_pct"] is None
    data = pd.DataFrame([dict(curve="force", step=0, time_seconds=0., seed=0, count=10, sum=10., sum_squares=10., attempted_trials=10),
                         dict(curve="force", step=0, time_seconds=0., seed=1, count=1, sum=3., sum_squares=9., attempted_trials=1)])
    result = a.aggregate_curves(data).iloc[0]
    assert result["mean"] == 2 and result.sd == 1 and result.contributing_trials == 11
    assert a.aggregate_curves(data, "seed_sd").iloc[0].sd == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize("length,failed,valid", [(1, True, 0), (7, True, 6), (12, True, 11), (28, True, 27), (28, False, 28)])
def test_failure_boundaries_and_undefined_samples(length, failed, valid):
    meta = metadata()
    meta["prescribed_steps"] = 28
    data, trial = trial_data([.05]*length, failed)
    result, *_ = a.shared.trial_metrics(data, trial, meta)
    assert result["valid_steps"] == valid and result["S_track"] == valid/28
    assert result["failure_rate"] == float(failed)
    assert result["ee_error_m"] is None if valid == 0 else result["ee_error_m"] == pytest.approx(.05)
    if failed:
        assert all(not e["recovered"] for e in result["events"])


def test_comparison_notebook_reloads_cached_helper(tmp_path, monkeypatch):
    fixture(tmp_path/"raw", seeds=(0,), loads=("loaded",))
    a.summarize(config(tmp_path))
    cfg = dict(summary_roots=[str(tmp_path/"summaries")], output_dir=str(tmp_path/"comparison"),
               method_order=["b1z1_pact"], focused_methods=["b1z1_pact"])
    # Simulate a notebook kernel still holding the obsolete plotting function.
    monkeypatch.setattr(a, "compare", lambda _: pytest.fail("Stale helper used"))
    nb = json.loads((ROOT/"notebooks/exp2_compare_approaches.ipynb").read_text())
    source = next("".join(c["source"]) for c in nb["cells"] if "analysis.compare(CONFIG)" in "".join(c["source"]))
    namespace = dict(analysis=a, CONFIG=cfg, Path=Path, display=lambda _: None)
    exec(compile(source, "comparison-cell", "exec"), namespace)
    assert len(namespace["figure"].axes) == 6


def test_comparison_missing_methods_and_protocols(tmp_path):
    fixture(tmp_path/"raw", loads=("loaded",))
    a.summarize(config(tmp_path))
    cfg = dict(summary_roots=[str(tmp_path/"summaries")], output_dir=str(tmp_path/"comparison"),
               method_order=["missing", "b1z1_pact"], latex_method_names={"b1z1_pact": r"\PILOT{}"},
               focused_methods=["b1z1_pact", "missing"], display_names={"b1z1_pact":"PILOT"},
               figure_title="Custom task comparison")
    tables, fig, diagnostics = a.compare(cfg)
    assert len(fig.axes) == 6
    assert fig._suptitle.get_text() == "Custom task comparison"
    for row in range(2):
        for col, profile in enumerate(a.PROFILES):
            ax = fig.axes[row*3+col]
            labels = [t.get_text() for t in ax.texts]
            axis = "Fz" if profile == "carrying" else "Fx"
            sign = "+" if profile == "pulling" else "-"
            expected = [f"{axis},{sign}{force}N" for force in (25,45,65)]+[f"{axis},0N"]
            assert labels == (expected if row == 0 else [])
            if row == 0:
                assert all(t.xy[1] == 1 and t.get_position()[1] > 0 and t.get_rotation() == 0 for t in ax.texts)
            assert not any("n=" in text for text in labels)
            assert sum(line.get_color() == "red" for line in ax.lines) == 4
    assert "—" in tables["response"].loc["missing"].to_string()
    assert any("Missing missing" in d for d in diagnostics)
    assert (tmp_path/"comparison/task_response.pdf").stat().st_size > 1000
    assert "\\begin{table*}" in (tmp_path/"comparison/main.tex").read_text()
    assert r"\PILOT{}" in (tmp_path/"comparison/main.tex").read_text()
    assert list(tables["main"].index) == ["missing", "PILOT"]
    assert "\\begin{table}" in (tmp_path/"comparison/response.tex").read_text()
    raw = pd.read_csv(tmp_path/"comparison/main.csv")
    assert raw.loc[(raw.metric == "failure_rate") & (raw.task == "b1z1_pact"), "mean"].iloc[0] == .5
    fixture(tmp_path/"raw", task="b1z1_ppo_pos", loads=("loaded",))
    a.summarize(config(tmp_path, "b1z1_ppo_pos"))
    path = tmp_path/"summaries/b1z1_ppo_pos/summary.json"
    s = a.shared.read_json(path)
    s["per_seed"] = [r for r in s["per_seed"] if r["seed"] != 2]
    a.shared.write_json(path, s)
    with pytest.raises(ValueError, match="seed coverage"):
        a.compare(cfg)
    s["protocol_sha256"] = "wrong"
    a.shared.write_json(path, s)
    with pytest.raises(ValueError, match="protocol"):
        a.compare(cfg)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_incompatible_runs_and_reset_safe_smoothness(tmp_path):
    fixture(tmp_path/"raw", seeds=(0,), loads=("loaded",))
    path = next((tmp_path/"raw").rglob("metadata.json"))
    meta = a.shared.read_json(path)
    meta["evaluation"]["ee_tolerance_m"] = .2
    a.shared.write_json(path, meta)
    with pytest.raises(ValueError, match="incompatible"):
        a.summarize(config(tmp_path))
    value, n = a.shared.acceleration_rms(np.array([[0.,0,0],[2.,0,0],[100.,0,0],[104.,0,0]]),
                                        np.array([0.,1.,4.,5.]), np.array([0,1,4,5]), np.ones(4,bool), 1.)
    assert n == 2 and value == pytest.approx(np.sqrt(10))


def test_notebooks_execute_without_simulator(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    fixture(tmp_path/"raw")
    # Different nominal gains are intentional; portable protocol checks allow them.
    for task, gain in (("b1z1_unifp_original", 20.), ("b1z1_ppo_pos", 30.)):
        fixture(tmp_path/"raw", task=task)
        for path in (tmp_path/"raw"/task).rglob("metadata.json"):
            meta = a.shared.read_json(path)
            meta["resolved_environment"]["control"]["stiffness"] = gain
            a.shared.write_json(path, meta)
        a.summarize(config(tmp_path, task))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    configs = [("exp2_summarize_approach", "EXP2_ANALYSIS_CONFIG", config(tmp_path)),
               ("exp2_compare_approaches", "EXP2_COMPARE_CONFIG", dict(summary_roots=[str(tmp_path/"summaries")], output_dir=str(tmp_path/"comparison")))]
    for name, variable, cfg in configs:
        monkeypatch.setenv(variable, json.dumps(cfg))
        nb = nbformat.read(ROOT/f"notebooks/{name}.ipynb", as_version=4)
        nb.cells.append(nbformat.v4.new_code_cell("assert 'torch' not in sys.modules and 'isaaclab' not in sys.modules"))
        nbclient.NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata":{"path":str(ROOT)}}).execute()
        nbformat.write(nb, tmp_path/f"{name}.ipynb")
    for name in ("main.csv", "main.tex", "physical_limits.csv", "response.tex", "task_response.csv", "task_response.pdf", "task_response.png"):
        assert (tmp_path/"comparison"/name).stat().st_size > 0
    curves = pd.read_csv(tmp_path/"comparison/task_response.csv")
    assert curves.task.nunique() == 3 and curves.profile.nunique() == 3

