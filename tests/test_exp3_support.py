"""Small offline fixtures: no CUDA, simulator, checkpoints or BARD required."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"notebooks"))
import exp3_support as s
import exp3_representation as r
from test_exp2_analysis import fixture as exp2_fixture


def test_passive_damping_is_not_friction_or_explicit_pd():
    row = dict(joint_velocity_t=np.array([2., -3.]), mechanics_joint_damping=np.array([.4]),
               mechanics_joint_friction=np.array([.1]), mechanics_joint_stiffness=np.array([1.]))
    bias, notes = s.passive_bias(row)
    np.testing.assert_allclose(bias, [.8, -1.2])
    assert any("constraint-dependent" in n for n in notes)
    assert any("stiffness" in n for n in notes)
    mass = np.eye(8)
    mass[6:, 6:] += np.diag([.2, .2])
    full_bias = np.r_[np.zeros(6), bias]
    velocity = np.r_[np.zeros(6), row["joint_velocity_t"]]
    result = r.dynamics_step(mass, full_bias, velocity, np.array([2., 3.]),
        np.zeros((1, 6, 8)), np.zeros((6, 8)), np.zeros((6, 8)), np.zeros((1, 3)),
        np.zeros(6), np.zeros(6), .1)
    np.testing.assert_allclose(result[6:], row["joint_velocity_t"] + .1*(np.array([2., 3.])-bias)/1.2)


def test_torque_squares_and_cancellation():
    row = dict(pd_weighted_motor_mean_square=np.array([5., 25.]),
               ff_weighted_motor_mean_square=np.array([1., 4.]),
               executed_torque_mean_square=np.array([4., 9.]),
               opposing_channel_cancellation_mean=np.array([1., 2.]))
    values = s.channel_values(row, ["hip", "z1_elbow"])
    assert values["arm_pd_weighted_motor_ms"] == 25
    assert values["leg_pd_weighted_motor_ms"] == 5  # NOT squared mean of [1,3] == 4.
    assert values["arm_cancellation_nm"] == 2


def test_support_uses_mean_of_cross_products():
    row = dict(grf_raw_world_interval_mean=np.array([[0., 0., 4.]]),
               support_moment_world=np.array([1., 2., 3.]),
               all_contact_moment_world_interval_mean=np.array([4., 5., 6.]),
               body_contact_forces_raw_world_interval_mean=np.array([[0., 0., 4.], [1., 0., 0.]]))
    values = s.support_values(row, ["foot"])
    assert values["foot_support_my_nm"] == 2
    assert values["all_support_mz_nm"] == 6
    assert values["all_support_fx_n"] == 1
    assert "foot_support_mx_nm" not in s.support_values({}, [])


def test_checkpoint_hierarchy():
    rows = [dict(task="p", checkpoint=c, seed=seed, trial=t, value=v)
            for c, seeds, v in [("a", range(3), 1.), ("b", range(1), 3.)]
            for seed in seeds for t in range(2)]
    result = s.hierarchy(pd.DataFrame(rows), ["task"]).iloc[0]
    assert result["mean"] == 2
    assert result.sd == pytest.approx(np.sqrt(2))
    assert result.variability == "checkpoint_sd_independence_unverified"
    single = s.hierarchy(pd.DataFrame(rows[:2]), ["task"]).iloc[0]
    assert single.sd is None or pd.isna(single.sd)
    assert single.variability == "evaluation_seed_sd"
    known = pd.DataFrame(rows)
    known["training_seed"] = known.checkpoint.map({"a": 1, "b": 2})
    assert s.hierarchy(known, ["task"]).iloc[0].variability == "training_seed_sd"


def test_urdf_fk(tmp_path):
    file = tmp_path/"arm.urdf"
    file.write_text('<robot name="a"><joint name="z1_joint" type="revolute"><parent link="base"/>'
        '<child link="arm"/><axis xyz="0 0 1"/></joint><joint name="fixed" type="fixed">'
        '<parent link="arm"/><child link="ee"/><origin xyz="1 0 0"/></joint></robot>')
    fk = s.RecordedFK(file, "base", "ee")
    np.testing.assert_allclose(fk([np.pi/2], ["z1_joint"]), [0., 1., 0.], atol=1e-12)
    with pytest.raises(ValueError, match="Missing commanded"):
        fk([], [])
    # The measured trunk can be a fixed sibling of the arm under the root.
    file.write_text(file.read_text().replace('</robot>', '<joint name="mount" type="fixed">'
        '<parent link="base"/><child link="trunk"/><origin xyz="0 0 0.2"/></joint></robot>'))
    np.testing.assert_allclose(s.RecordedFK(file, "trunk", "ee")([0], ["z1_joint"]), [1, 0, -.2])


def fixture(root):
    exp2_fixture(root, seeds=(0,), loads=("loaded",))
    for path in root.rglob("metadata.json"):
        meta = s.shared.read_json(path)
        meta["dof_names"] = ["hip", "z1_joint"]
        meta["resolved_environment"]["control"] = dict(stiffness={"joint": 10}, damping={"joint": 1})
        meta["resolved_environment"]["asset"] = dict(foot_name=["foot"], base_name="base", gripper_name="ee")
        s.shared.write_json(path, meta)
        manifest_path = path.parent/"records.json"
        manifest = s.shared.read_json(manifest_path)
        for chunk in manifest["chunks"]:
            file = path.parent/chunk["file"]
            with np.load(file) as data:
                values = dict(data)
            n = chunk["rows"]
            values["task_stage_interval"] = s.exp2.stage_ids(values["time_seconds"], meta["evaluation"])
            for field in ("pd_weighted_motor_mean_square", "ff_weighted_motor_mean_square", "executed_torque_mean_square"):
                values[field] = np.ones((n, 2))*4
            values["opposing_channel_cancellation_mean"] = np.ones((n, 2))
            values["support_moment_world"] = np.ones((n, 3))*2
            values["all_contact_moment_world_interval_mean"] = np.ones((n, 3))*3
            values["grf_raw_world_interval_mean"] = np.ones((n, 1, 3))*4
            values["body_contact_forces_raw_world_interval_mean"] = np.ones((n, 1, 3))*4
            manifest["fields"].update({k:dict(shape=list(v.shape[1:]), dtype=str(v.dtype)) for k,v in values.items()})
            np.savez_compressed(file, **values)
        s.shared.write_json(manifest_path, manifest)


def test_exp2_analysis_outputs_and_failure_alignment(tmp_path):
    fixture(tmp_path/"raw")
    result = s.analyze(dict(output_dir=str(tmp_path/"out"), exp2_roots=[str(tmp_path/"raw")], evaluation_seeds=[0]))
    out = tmp_path/"out/action_support"
    assert len(list(out.glob("*.pdf"))) >= 3
    curves = pd.read_csv(out/"curves.csv")
    error = curves[(curves.profile == "pushing") & (curves.metric == "ee_error_m")]
    assert error.loc[error.step == 7, "surviving_trials"].item() == 1
    assert error.loc[error.step == 7, "time_seconds"].item() == 8
    forces = curves[(curves.profile == "pushing") & (curves.metric == "load_n")]
    assert forces.loc[forces.step == 7, "mean"].item() == -25
    assert forces.loc[forces.step == 7, "time_seconds"].item() == 7
    stages = pd.read_csv(out/"trial_stages.csv")
    assert (stages.loc[stages.metric == "arm_pd_weighted_motor_rms_nm", "value"] == 2).all()
    assert any("Missing Exp2 method" in x for x in result["diagnostics"])
    assert (out/"COMPLETE").exists()


def test_per_stratum_coverage(tmp_path):
    from test_exp3_representation import fixture, config
    fixture(tmp_path/"raw")
    runs, _ = r.discover(config(tmp_path))
    first, coverage = r.select_manifest(runs, None, 4, per_stratum=3, minimum=4)
    second, _ = r.select_manifest(runs, None, 4, per_stratum=3, minimum=4)
    assert first == second
    assert all(c["selected"] == min(3, c["available"]) for c in coverage)
    assert not any(c["sufficient"] for c in coverage)


def test_reconstruction_requires_native_contract(tmp_path):
    import copy
    from test_exp3_representation import fixture, config
    fixture(tmp_path/"raw")
    runs, _ = r.discover(config(tmp_path))
    run = runs[0]
    manifest, _ = r.select_manifest([run], 4, 0)
    data = r.selected_data(run, manifest)
    assert r.reconstruct_inputs(data, run["meta"], run["meta"]) is data
    changed = copy.deepcopy(run["meta"])
    changed["resolved_environment"]["env"]["num_policy_actions"] = 2
    with pytest.raises(ValueError, match="Unreconstructable"):
        r.reconstruct_inputs(data, run["meta"], changed)
    changed = copy.deepcopy(data)
    changed[0]["history_t"][-1] += 1
    with pytest.raises(ValueError, match="causal"):
        r.reconstruct_inputs(changed, run["meta"], run["meta"])
