"""Metadata-driven batch discovery and CPU notebook execution."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from experiment_batch import plan_summaries


def record(root, task, experiment=1, complete=True, checkpoint="hash"):
    directory = root / f"arbitrary-{len(list(root.glob('*')))}"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps(dict(
        task=task, experiment=experiment, evaluation_seed=0,
        checkpoint_sha256=checkpoint, arguments={"loading": "loaded"})))
    if complete:
        (directory / "COMPLETE").touch()


def test_discovery_selection_isolation_and_legacy(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    record(raw, "method_b")
    record(raw, "method_a")
    record(raw, "unfinished", complete=False)
    record(raw, "exp2_only", experiment=2)
    cfg = dict(input_roots=[str(raw)], output_dir=str(tmp_path / "out"),
               tasks=None, task_options={"method_a": {"display_name": "A"}})
    before = copy.deepcopy(cfg)
    plans, inventory = plan_summaries(cfg, 1)
    assert cfg == before
    assert [p["task"] for p in plans] == ["method_a", "method_b"]
    assert plans[0]["display_name"] == "A"
    assert plans[0]["output_dir"] == str(tmp_path / "out/method_a")
    assert inventory[0]["evaluation_seeds"] == [0]
    plans, _ = plan_summaries({**cfg, "tasks": ["method_b", "method_a"]}, 1)
    assert [p["task"] for p in plans] == ["method_b", "method_a"]
    plans, _ = plan_summaries({**cfg, "task": "method_a"}, 1)
    assert plans[0]["output_dir"] == cfg["output_dir"]
    with pytest.raises(ValueError, match="requested tasks"):
        plan_summaries({**cfg, "tasks": ["unfinished"]}, 1)
    with pytest.raises(ValueError, match="unique"):
        plan_summaries({**cfg, "tasks": ["method_a", "method_a"]}, 1)
    record(raw, "method_a", checkpoint="second")
    plans, _ = plan_summaries({**cfg, "tasks": ["method_a"],
        "task_options": {"method_a": {"checkpoint_sha256": "second"}}}, 1)
    assert len(plans[0]["input_roots"]) == 1


@pytest.mark.parametrize("experiment", [1, 2])
def test_batch_notebook_two_methods(tmp_path, monkeypatch, experiment):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    from test_exp1_analysis import make_fixture
    from test_exp2_analysis import fixture
    tasks = ["b1z1_pact", "b1z1_ppo_pos"]
    for task in tasks:
        if experiment == 1:
            make_fixture(tmp_path / "raw", task)
        else:
            fixture(tmp_path / "raw", task, seeds=(0,), loads=("loaded",))
    cfg = dict(input_roots=[str(tmp_path / "raw")], task=None,
               tasks=None if experiment == 1 else tasks[::-1],
               output_dir=str(tmp_path / "summaries"), loading=["loaded"],
               expected_seeds=[0, 1, 2], histogram_edges=[-.1, 0., .1])
    if experiment == 1:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg))
        monkeypatch.setenv("EXP1_ANALYSIS_CONFIG", str(path))
    else:
        monkeypatch.setenv("EXP2_ANALYSIS_CONFIG", json.dumps(cfg))
    nb = nbformat.read(ROOT / f"notebooks/exp{experiment}_summarize_approach.ipynb", as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(
        "assert len(summaries) == 2\nassert 'torch' not in sys.modules and 'isaaclab' not in sys.modules"))
    nbclient.NotebookClient(nb, timeout=180, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}}).execute()
    for task in tasks:
        folder = tmp_path / "summaries" / task
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["task"] == task and summary["status"] == "complete"
        assert (folder / "COMPLETE").exists() and (folder / "summary.csv").exists()
