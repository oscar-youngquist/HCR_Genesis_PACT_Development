"""CPU fixtures, optional real Torch constructors, and a notebook saved-prediction audit."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"notebooks"))
import exp3_representation as a
from test_exp1_analysis import metadata, trial_data


def fixture(root, tasks=("b1z1_pact",), seeds=(0,1)):
    for task in tasks:
        ckpt = root/(task+".pt")
        root.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(("synthetic saved-prediction audit: "+task).encode())
        for seed in seeds:
            for condition in ("nominal","ee-disturbance","combined-disturbance"):
                for load in ("in","out"):
                    path = root/task/f"{seed}-{condition}-{load}"
                    path.mkdir(parents=True)
                    m = metadata(task,seed,condition)
                    m.update(checkpoint=str(ckpt),checkpoint_sha256=a.file_hash(ckpt),runner="B1Z1PACTRunner",
                             dof_names=["leg","z1_arm"],body_names=["base","ee"],base_index=0,gripper_index=1,
                             resolved_training=dict(policy=dict(cenet_latent_dim=3,cenet_enc_layers=[8],actor_layers=[8],critic_layers=[8],
                                 explicit_decoder_layers=[8],force_decoder_layers=[8],grf_decoder_layers=[8]),runner={}))
                    m["arguments"]["load_range"] = load
                    m["prescribed_steps"] = 8
                    m["resolved_environment"]["env"].update(num_observations=4,num_obs_hist=2,num_actions=2,num_policy_actions=4,num_explicit_recon_obs=23)
                    if task == "b1z1_ppo_pos":
                        m["resolved_environment"]["env"]["num_policy_actions"] = 2
                    if task == "b1z1_unifp_original":
                        m["runner"] = "UniFPOriginalRunner"
                    m["resolved_environment"]["normalization"] = dict(obs_scales=dict(ee_force=.1,base_force=.01,grf=.001))
                    m["resolved_environment"]["asset"] = dict(foot_name=["FR","FL","RR","RL"])
                    arrays, trials, initial = [], [], []
                    for tid,length in enumerate((8,5)):
                        d,t=trial_data([.01]*length,tid==1,tid,condition,seed)
                        t["motion_mode"] = "stationary" if tid==0 else "locomotion"
                        d["phase"] = np.where(d["step"]<3,0,1)
                        obs=np.column_stack([np.arange(length),np.ones((length,3))]).astype(np.float32)
                        d.update(obs_t=obs,history_t=np.concatenate((np.vstack((obs[0],obs[:-1])),obs),axis=-1),
                            privileged_t=np.zeros((length,3),np.float32),action_raw=np.zeros((length,4),np.float32),
                            ee_force_world=np.tile([5.,0.,0.],(length,1)),base_force_world=np.zeros((length,3)),base_moment_world=np.zeros((length,3)),
                            prediction_z=np.ones((length,3),np.float32),prediction_ee_force=np.tile([.6,0,0],(length,1)),
                            prediction_base_wrench=np.zeros((length,6)),prediction_grf=np.ones((length,12))*.001,
                            prediction_grf_torque_input_nm=np.zeros((length,2)),grf_raw_world_interval_mean=np.ones((length,4,3)),
                            joint_position_t=np.zeros((length,2)),joint_velocity_t=np.zeros((length,2)),joint_velocity=np.zeros((length,2)),
                            base_position_t=np.zeros((length,3)),base_quaternion_t=d["base_quaternion_xyzw"].copy(),
                            base_velocity_world_t=np.zeros((length,3)),base_angular_velocity_world_t=np.zeros((length,3)),
                            base_velocity_world=np.zeros((length,3)),base_angular_velocity_world=np.zeros((length,3)),
                            time_seconds=np.arange(length,dtype=float),executed_torque_interval_mean=np.zeros((length,2)))
                        d["ee_force_world"][:3]=0
                        if condition == "nominal":
                            d["ee_force_world"][:]=0
                        arrays.append(d); trials.append(t)
                        initial.append(dict(trial_id=tid,body_masses_kg=[1.,1.],body_inertias_kg_m2=[np.eye(3).flatten().tolist()]*2,
                                            body_local_com_pose=[[0,0,0,1,0,0,0]]*2))
                    d={k:np.concatenate([v[k] for v in arrays]) for k in arrays[0]}
                    if task == "b1z1_unifp_original":
                        d["prediction_explicit"] = np.zeros((len(d["step"]), 12))
                        d["prediction_explicit"][:, 6] = .6
                        for key in ("prediction_ee_force", "prediction_base_wrench", "prediction_grf", "prediction_grf_torque_input_nm"):
                            d.pop(key)
                    idx=np.argsort(d["step"],kind="stable"); d={k:v[idx] for k,v in d.items()}
                    parts=[]
                    for n,start in enumerate(range(0,len(idx),4)):
                        filename=f"transitions-{n}.npz"; part={k:v[start:start+4] for k,v in d.items()}
                        np.savez_compressed(path/filename,**part); parts.append(dict(file=filename,rows=len(part["step"])))
                    a.shared.write_json(path/"records.json",dict(fields={k:dict(shape=list(v.shape[1:]),dtype=str(v.dtype)) for k,v in d.items()},chunks=parts))
                    a.shared.write_json(path/"metadata.json",m);a.shared.write_json(path/"trials.json",trials)
                    (path/"initial_states.jsonl").write_text("\n".join(json.dumps(x) for x in initial))
                    (path/"COMPLETE").write_text("ok")


def config(tmp):
    return dict(recording_roots=[str(tmp/"raw")],methods={"PILOT":dict(task="b1z1_pact")},evaluation_seeds=[0,1],
        conditions=["nominal","ee-disturbance","combined-disturbance"],sampling_budget=48,sampling_seed=3,
        output_dir=str(tmp/"output"),device="cpu",dynamics=False,recorded_native_only=True)


def remove_fields(root, names):
    for manifest in root.rglob('records.json'):
        records=json.loads(manifest.read_text())
        for name in names: records['fields'].pop(name,None)
        for chunk in records['chunks']:
            path=manifest.parent/chunk['file']
            with np.load(path) as saved: arrays={k:saved[k] for k in saved.files if k not in names}
            np.savez_compressed(path,**arrays)
        manifest.write_text(json.dumps(records))


def test_stored_metrics_need_no_checkpoint_history_or_bard(tmp_path,monkeypatch):
    fixture(tmp_path/'raw',seeds=(0,))
    remove_fields(tmp_path/'raw',['history_t','obs_t'])
    for path in (tmp_path/'raw').glob('*.pt'): path.unlink()
    for path in (tmp_path/'raw').rglob('initial_states.jsonl'): path.unlink()
    monkeypatch.setattr(a,'build_model',lambda *args: pytest.fail('Unneeded model'))
    monkeypatch.setattr(a,'OfflineBard',lambda *args,**kwargs: pytest.fail('Unneeded BARD'))
    cfg=config(tmp_path); cfg.update(evaluation_seeds=[0],recorded_native_only=False,dynamics=False)
    result=a.run_analysis(cfg)
    assert result['selection']['selected']>0
    assert any(r['metric']=='ee_force_rmse' for r in result['table_summary'])
    assert any(r['metric']=='grf_rmse' for r in result['table_summary'])


def test_current_velocity_is_not_successor():
    row=dict(prediction_base_velocity=np.ones(3),base_velocity_world=np.zeros(3),base_angular_velocity_world=np.zeros(3),joint_velocity=np.zeros(2))
    assert a.stored_successor_error(row,'decoded') is None
    row['prediction_next_velocity_decoded']=np.ones(8)
    np.testing.assert_array_equal(a.stored_successor_error(row,'decoded'),np.ones(8))


def test_recorded_dynamics_avoids_bard_initialization(tmp_path,monkeypatch):
    fixture(tmp_path/'raw',seeds=(0,))
    terms=dict(mass_matrix=np.eye(8),bias=np.zeros(8),foot_jacobians=np.zeros((4,6,8)),
               ee_jacobian=np.eye(6,8),base_jacobian=np.eye(6,8),ee_lever=np.zeros(3),base_lever=np.zeros(3))
    for manifest in (tmp_path/'raw').rglob('records.json'):
        records=json.loads(manifest.read_text())
        for chunk in records['chunks']:
            path=manifest.parent/chunk['file']
            with np.load(path) as saved: data=dict(saved)
            for key,value in terms.items(): data['dynamics_'+key]=np.broadcast_to(value,(len(data['step']),*value.shape))
            np.savez_compressed(path,**data)
        for key,value in terms.items(): records['fields']['dynamics_'+key]=dict(shape=list(value.shape),dtype='float64')
        manifest.write_text(json.dumps(records))
    monkeypatch.setattr(a,'OfflineBard',lambda *args,**kwargs: pytest.fail('Recorded terms should bypass BARD'))
    cfg=config(tmp_path); cfg.update(evaluation_seeds=[0],dynamics=True)
    result=a.run_analysis(cfg)
    assert result['prediction_sources']['stored:dynamics_terms']==result['selection']['selected']
    assert {r['variant'] for r in result['table_summary'] if r['metric']=='base_linear_rmse'}=={'decoded','measured_loading','constant_velocity'}


def test_sampling_causal_targets_and_missing_histories(tmp_path):
    fixture(tmp_path/"raw"); runs,_=a.discover(config(tmp_path))
    manifest,balance=a.select_manifest(runs,48,3)
    again,_=a.select_manifest(list(reversed(runs)),48,3)
    assert manifest==again
    for load in ("in","out"):
        cohort=[r for r in balance if r["stratum"][0]==load]
        counts=[r["selected"] for r in cohort if r["selected"]<r["available"]]
        assert max(counts)-min(counts)<=1  # Exhausted sparse strata cannot supply more rows.
        assert sum(r["selected"] for r in cohort)==48
    assert all(e["step"]>=1 for e in manifest)
    run=runs[0]
    all_entries=[e for _,e in a.candidates(run)]
    rows=a.selected_data(run,all_entries)
    boundary=next(r for r in rows if r["step"]==3)
    np.testing.assert_array_equal(boundary["history_ee_force_world"],[0,0,0])
    np.testing.assert_array_equal(boundary["ee_force_world"],[5,0,0])
    assert not any(r["physical_failure"] for r in rows)
    assert rows[0]["history_t"].shape==(8,)


def test_normalization_rotation_missing_outputs():
    q=np.array([0.,0.,np.sqrt(.5),np.sqrt(.5)])
    m=metadata(); m["runner"]="UniFPOriginalRunner"
    m["resolved_environment"]["normalization"]=dict(obs_scales=dict(ee_force=.1,base_force=.01,grf=.001))
    row=dict(base_quaternion_t=q,base_quaternion_xyzw=np.array([0.,0.,0.,1.]))
    p=a.physical_predictions(dict(prediction_explicit=np.array([0.]*6+[.2,0,0,.03,0,0])),m,row)
    np.testing.assert_allclose(p["ee_force"],[0,2,0],atol=1e-14)
    np.testing.assert_allclose(p["base_force"],[0,3,0],atol=1e-14)
    assert "base_moment" not in p and "grf" not in p
    m["runner"]="B1Z1PACTRunner";m["resolved_environment"]["env"].update(num_actions=2,num_policy_actions=4)
    p=a.physical_predictions(dict(prediction_ee_force=np.array([.2,0,0]),prediction_base_wrench=np.array([.03,0,0,0,.04,0]),prediction_grf=np.tile([.001,0,0],4)),m,row)
    np.testing.assert_allclose(p["base_moment"],[-4,0,0],atol=1e-14)
    np.testing.assert_allclose(p["grf"],[[1,0,0]]*4)


def test_synthetic_dynamics_and_reference_point():
    n=8;mass=np.eye(n)*2;bias=np.zeros(n);pre=np.zeros(n)
    feet=np.zeros((4,3,n));feet[0,:,:3]=np.eye(3)
    ee=np.zeros((6,n));ee[:,:6]=np.eye(6);base=ee.copy()
    out=a.dynamics_step(mass,bias,pre,np.array([2.,4.]),feet,ee,base,np.ones((4,3)),np.zeros(6),np.zeros(6),.1)
    np.testing.assert_allclose(out,np.array([.05,.05,.05,0,0,0,.1,.2]))
    row=dict(base_velocity_world_t=np.zeros(3),base_angular_velocity_world_t=np.zeros(3),joint_velocity_t=np.zeros(2),
        base_velocity_world=np.zeros(3),base_angular_velocity_world=np.zeros(3),joint_velocity=np.zeros(2),executed_torque_interval_mean=np.zeros(2))
    fixed=dict(mass_matrix=np.eye(n),bias=bias,foot_jacobians=feet,ee_jacobian=ee,base_jacobian=base,base_lever=np.array([0,1,0]),ee_lever=np.zeros(3))
    forces=dict(ee_force=np.zeros(3),base_force=np.array([2,0,0]),base_moment=np.zeros(3),grf=np.zeros((4,3)))
    error=a.velocity_errors(row,forces,fixed,.1)
    assert error[0]==.2 and error[5]==-.2  # r x F, shifted exactly once.


def test_aggregation_and_parity():
    rows=[]
    for seed,values in [(0,[(0,1.,10),(1,3.,1)]),(1,[(0,4.,1)])]:
        for tid,value,count in values:
            for _ in range(count):
                rows.append(dict(method="P",benchmark="native",variant="decoded",load_range="in",condition="nominal",motion="all",phase="all",
                    load_timing="sustained",metric="ee_force_rmse",axis="all",evaluation_seed=seed,source="P",run="r",trial_id=tid,mse=value**2,bias=value))
    seeds,summary=a.aggregate(rows)
    assert [r["mean"] for r in seeds]==[2,4]
    assert summary[0]["mean"]==3 and summary[0]["sd"]==pytest.approx(np.sqrt(2))
    assert summary[0]["contributing_samples"]==12
    d=[dict(prediction_z=np.ones(3))]
    assert a.parity(dict(prediction_z=np.ones((1,3))),d)==["prediction_z"]
    with pytest.raises(ValueError,match="parity"):
        a.parity(dict(prediction_z=np.zeros((1,3))),d)


def test_saved_native_run_and_contract_separation(tmp_path):
    fixture(tmp_path/"raw"); cfg=config(tmp_path)
    result=a.run_analysis(cfg)
    assert result["recorded_native_only"]
    assert {r["benchmark"] for r in result["table_summary"]}=={"native"}
    assert not any(r["variant"]=="decoded" and r["metric"]=="base_linear_rmse" for r in result["table_summary"])
    assert (Path(result["output_dir"])/"native_in_table.tex").exists()
    assert "NaN" not in (Path(result["output_dir"])/"summary.json").read_text()
    runs,_=a.discover(cfg);m=runs[0]["meta"];pos=copy.deepcopy(m)
    pos["resolved_environment"]["env"]["num_policy_actions"]=2
    assert a.input_contract(m)!=a.input_contract(pos)
    cfg["recorded_native_only"]=False
    assert any("disabled" in r["reason"] for r in result["unavailable"])


def test_native_transition_dispatch(tmp_path, monkeypatch):
    fixture(tmp_path/"raw", tasks=("b1z1_pact","b1z1_pact_ab4"), seeds=(0,))
    remove_fields(tmp_path/'raw',['prediction_grf'])
    cfg=config(tmp_path)
    cfg.update(recorded_native_only=False, evaluation_seeds=[0], sampling_budget=24, inference_batch_size=3)
    cfg["methods"]["ab4"] = dict(task="b1z1_pact_ab4")
    # Independent deterministic toy heads test replay routing without Torch/Jupyter.
    seen = {}
    def infer(model, data, device):
        assert len(data) <= 3
        assert all(row["entry"]["source"] == model for row in data)
        seen.setdefault(model, set()).update(row["entry"]["transition_id"] for row in data)
        n=len(data)
        return dict(prediction_z=np.ones((n,3)), prediction_ee_force=np.tile([.6,0,0],(n,1)),
                    prediction_base_wrench=np.zeros((n,6)), prediction_grf=np.ones((n,12))*.001)
    monkeypatch.setattr(a,"build_model",lambda run, device: run["name"])
    monkeypatch.setattr(a,"infer",infer)
    monkeypatch.setattr(a,"input_contract",lambda *args: pytest.fail("No compatibility grouping in native analysis"))
    monkeypatch.setattr(a,"select_manifest",lambda *args, **kwargs: pytest.fail("Full-data default must not sample"))
    result=a._run_analysis(cfg)
    assert seen["PILOT"].isdisjoint(seen["ab4"]) and seen["PILOT"]
    assert result["selection"]["selected"] == sum(map(len, seen.values()))
    assert all(r["benchmark"] == "native" for r in result["table_summary"])


def test_torch_checkpoint_native_replay(tmp_path):
    torch=pytest.importorskip("torch")
    fixture(tmp_path/"raw",seeds=(0,)); runs,_=a.discover(config(tmp_path)); run=runs[0]
    from rsl_rl.modules.actor_critic_b1z1_pact import ActorCriticB1Z1PACT
    actor=ActorCriticB1Z1PACT(4,3,2,8,latent_dim=3,context_layers=[8],actor_layers=[8],critic_layers=[8],
        explicit_decoder_layers=[8],force_decoder_layers=[8],grf_decoder_layers=[8]).eval()
    torch.save(dict(model_state_dict=actor.state_dict()),run["checkpoint"])
    with pytest.raises(ValueError,match='mismatched fallback checkpoint'):
        a.build_model(run,'cpu')
    run['meta']['checkpoint_sha256']=a.file_hash(run['checkpoint'])
    model=a.build_model(run,"cpu")
    entries=[e for _,e in a.candidates(run)][:2];d=a.selected_data(run,entries)
    context = actor.decode_context(actor.context_encoder.forward_inf(torch.zeros(1,8)))
    torque_width = actor.physics_decoder.grf[0].in_features - context["z"].shape[-1] - context["explicit_condition"].shape[-1]
    for row in d:
        row["prediction_grf_torque_input_nm"] = np.zeros(torque_width, np.float32)
    expected=a.infer(actor,d,"cpu")
    for i,row in enumerate(d):
        for k,v in expected.items():row[k]=v[i]
    rng=torch.random.get_rng_state().clone(); before={k:v.clone() for k,v in model.state_dict().items()}
    actual=a.infer(model,d,"cpu")
    assert "prediction_grf" in a.parity(actual,d)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("include_exp2", [False, True])
def test_notebook_cpu_saved_prediction_smoke(tmp_path,monkeypatch,include_exp2):
    nbformat=pytest.importorskip("nbformat");nbclient=pytest.importorskip("nbclient")
    fixture(tmp_path/"raw", tasks=("b1z1_pact", "b1z1_ppo_pos"))
    cfg = config(tmp_path)
    cfg["methods"] = None
    if include_exp2:
        from test_exp3_support import fixture as support_fixture
        support_fixture(tmp_path/"exp2")
    cfg["exp2_roots"] = [str(tmp_path/"exp2")]
    monkeypatch.setenv("EXP3_CONFIG",json.dumps(cfg))
    nb=nbformat.read(ROOT/"notebooks/exp3_representation_analysis.ipynb",as_version=4)
    if not include_exp2:
        nb.cells = [cell for cell in nb.cells if not cell.id.startswith("exp3-action")]
    nb.cells.append(nbformat.v4.new_code_cell("assert 'torch' not in sys.modules and 'isaaclab' not in sys.modules"))
    nbclient.NotebookClient(nb,timeout=180,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
    nbformat.write(nb,tmp_path/"executed_exp3.ipynb")
    for name in ("transition_manifest.json", "transition_manifest.jsonl", "per_seed.csv", "summary.csv", "native_in_table.tex", "COMPLETE"):
        assert (tmp_path/"output/representation/full_recorded_first"/name).stat().st_size>0
    assert (tmp_path/"output/action_support/COMPLETE").exists() == include_exp2


def test_method_discovery_and_selection(tmp_path):
    fixture(tmp_path/"raw", tasks=("b1z1_pact", "b1z1_ppo_pos"))
    cfg = config(tmp_path)
    cfg["methods"] = None
    resolved = a.resolve_methods(cfg)
    assert set(resolved["methods"]) == {"b1z1_pact", "b1z1_ppo_pos"}
    assert resolved["methods"]["b1z1_ppo_pos"]["display_name"] == "Pos"
    assert cfg["methods"] is None
    cfg["tasks"] = ["b1z1_ppo_pos"]
    selected = a.resolve_methods(cfg)
    assert list(selected["methods"]) == cfg["tasks"]
    assert a.resolve_methods(selected) == selected
    runs, _ = a.discover(selected)
    assert {r["meta"]["task"] for r in runs} == {"b1z1_ppo_pos"}
    assert all(r["checkpoint"].name == "b1z1_ppo_pos.pt" for r in runs)
    cfg["tasks"] = ["absent"]
    with pytest.raises(ValueError, match="No completed"):
        a.resolve_methods(cfg)


def test_discovery_excludes_incomplete_and_exp2(tmp_path):
    root = tmp_path / "raw"
    for task, experiment, complete in (("pact", 1, True), ("exp2", 2, True), ("partial", 1, False)):
        path = root / task
        path.mkdir(parents=True)
        a.shared.write_json(path / "metadata.json", dict(task=task, experiment=experiment,
            evaluation_seed=0, arguments=dict(condition="nominal")))
        if complete:
            (path / "COMPLETE").touch()
    cfg = config(tmp_path)
    cfg["methods"] = None
    assert list(a.resolve_methods(cfg)["methods"]) == ["pact"]


def test_full_mode_ignores_sampling_and_matches_in_memory(tmp_path, monkeypatch):
    fixture(tmp_path/"raw", seeds=(0,))
    cfg = config(tmp_path)
    run = a.discover(cfg)[0][0]
    cfg.update(recording_roots=[str(run["path"])], evaluation_seeds=[0], inference_batch_size=1)
    entries, _ = a.select_manifest([run], 10000, 0)
    data = a.selected_data(run, entries)
    rows = []
    for row in data:
        meta = run["meta"]
        physical = a.physical_predictions({k:v for k,v in row.items() if k.startswith("prediction_")}, meta, row)
        target = dict(ee_force=row["history_ee_force_world"], base_force=row["history_base_force_world"],
                      base_moment=row["history_base_moment_world"], grf=row["grf_raw_world_interval_mean"])
        timing = "new_load" if any(not np.allclose(row[k], row["history_"+k]) for k in a.LOADS) else "sustained"
        rows += a.metric_rows(row["entry"], "PILOT", "native", "decoded", {k:physical[k]-target[k] for k in physical},
                              meta["resolved_environment"]["asset"]["foot_name"], meta["dof_names"], timing)
        error = a.velocity_errors(row, None, None, meta["dt"])
        rows += a.metric_rows(row["entry"], "PILOT", "native", "constant_velocity", a.velocity_blocks(error, meta["dof_names"]),
                              [], meta["dof_names"], timing)
    monkeypatch.setattr(a, "select_manifest", lambda *args, **kwargs: pytest.fail("Default mode called balanced selection"))
    result = a.run_analysis(cfg)
    assert result["selection"]["available"] == result["selection"]["selected"] == len(entries)
    manifest_hash = result["selection"]["transitions_sha256"]
    _, expected = a.aggregate(rows)
    def ordered(values):
        return pd.DataFrame(values).sort_values(a.STRATA).reset_index(drop=True).sort_index(axis=1)
    pd.testing.assert_frame_equal(ordered(result["breakdown"]), ordered(expected), check_dtype=False)
    pooled = [dict(row, condition="nominal" if row["condition"] == "nominal" else "disturbed", motion="all", phase="all", load_timing="all") for row in rows]
    pd.testing.assert_frame_equal(ordered(result["table_summary"]), ordered(a.aggregate(pooled)[1]), check_dtype=False)
    cfg.update(sampling_budget=0, samples_per_stratum=0, sampling_seed=999, inference_batch_size=100)
    again = a.run_analysis(cfg)
    assert again["selection"]["transitions_sha256"] == manifest_hash
    pd.testing.assert_frame_equal(ordered(result["table_summary"]), ordered(again["table_summary"]), check_dtype=False)


def test_balanced_selection_is_per_checkpoint(tmp_path):
    fixture(tmp_path/"raw", tasks=("b1z1_pact", "b1z1_pact_second"), seeds=(0,))
    for path in (tmp_path/"raw/b1z1_pact_second").rglob("metadata.json"):
        meta = a.shared.read_json(path)
        meta["task"] = "b1z1_pact"
        a.shared.write_json(path, meta)
    cfg = config(tmp_path)
    cfg.update(evaluation_seeds=[0], balanced_resampling=True, sampling_budget=4, inference_batch_size=2,
        methods={name: dict(task="b1z1_pact", checkpoint=str(tmp_path/"raw"/(task+".pt")))
                 for name, task in (("first", "b1z1_pact"), ("second", "b1z1_pact_second"))})
    result = a.run_analysis(cfg)
    assert result["selection"]["selected"] == 16  # 4 per checkpoint and per range.
    manifest = [json.loads(line) for line in (Path(result["output_dir"])/"transition_manifest.jsonl").read_text().splitlines()]
    assert len({row["transition_id"] for row in manifest}) == 16
    for row in manifest:
        assert ("b1z1_pact_second" in row["run"]) == (row["source"] == "second")
    again = a.run_analysis(cfg)
    assert result["selection"]["transitions_sha256"] == again["selection"]["transitions_sha256"]
    assert all(c["selected"] <= c["available"] for c in result["coverage"])
    cfg["sampling_seed"] += 1
    assert a.run_analysis(cfg)["selection"]["transitions_sha256"] != result["selection"]["transitions_sha256"]


def test_mixed_native_families_missing_heads(tmp_path, monkeypatch):
    fixture(tmp_path/"raw", tasks=("b1z1_pact", "b1z1_ppo_pos", "b1z1_unifp_original"), seeds=(0,))
    cfg = config(tmp_path)
    cfg.update(methods=None, evaluation_seeds=[0], inference_batch_size=3)
    monkeypatch.setattr(a, "input_contract", lambda *args: pytest.fail("No cross-policy compatibility checks"))
    result = a.run_analysis(cfg)
    for task in ("b1z1_pact", "b1z1_ppo_pos", "b1z1_unifp_original"):
        assert any(r["task"] == task and r["metric"] == "ee_force_rmse" and r["mean"] is not None for r in result["comparative_summary"])
    unifp = [r for r in result["comparative_summary"] if r["task"] == "b1z1_unifp_original"]
    assert not any(r["metric"] in ("grf_rmse", "base_moment_rmse") for r in unifp)
    assert not any(r["variant"] == "decoded" and r["metric"] == "base_linear_rmse" for r in unifp)
    frame = pd.read_csv(Path(result["output_dir"])/"native_in_table.csv", header=[0, 1], index_col=0)
    assert set(frame.index) == {"PILOT", "Pos", "UniFP"}
    assert "—" in frame.loc["UniFP"].tolist()
    assert any(r["component"] == "grf" and r["count"] > 0 for r in result["unavailable"])


def test_failed_optional_inference_retains_stored_metrics(tmp_path, monkeypatch):
    fixture(tmp_path/"raw", seeds=(0,))
    remove_fields(tmp_path/"raw", ["prediction_grf"])
    cfg=config(tmp_path); cfg.update(evaluation_seeds=[0],recorded_native_only=False)
    monkeypatch.setattr(a,"build_model",lambda *args: object())
    monkeypatch.setattr(a,"infer",lambda *args: (_ for _ in ()).throw(RuntimeError("missing optional decoder")))
    result=a.run_analysis(cfg)
    assert any(r["metric"]=="ee_force_rmse" for r in result["table_summary"])
    assert not any(r["metric"]=="grf_rmse" for r in result["table_summary"])
    assert any(r["component"]=="optional decoder fallback" for r in result["unavailable"])


def test_streamed_run_merge_preserves_trial_weights():
    base = dict(method="P", benchmark="native", variant="decoded", load_range="in", condition="disturbed",
                motion="all", phase="all", load_timing="all", metric="ee_force_rmse", axis="all", evaluation_seed=0)
    seeds, summary = a.merge_seed_statistics([dict(base, mean=1., bias=-1., trials=10, samples=100),
                                             dict(base, mean=3., bias=3., trials=1, samples=1)])
    assert seeds[0]["mean"] == pytest.approx(13/11)
    assert seeds[0]["bias"] == pytest.approx(-7/11)
    assert summary[0]["contributing_samples"] == 101
    assert summary[0]["contributing_trials"] == 11


def test_offline_bard_import_never_imports_simulator():
    pytest.importorskip("torch")
    import subprocess
    # Fresh interpreter prevents an earlier import/cache from hiding side effects.
    code = r'''
import importlib.abc, os, sys
sys.path.insert(0, "notebooks")
class NoSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"legged_gym", "isaaclab", "isaacsim", "genesis"}:
            raise AssertionError("Unexpected simulator import: " + fullname)
sys.meta_path.insert(0, NoSimulator())
before = os.environ.get("SIMULATOR")
import exp3_representation as a
cls = a.offline_bard_class()
assert cls.__name__ == "BardB1Z1DynamicsBackend"
assert a.offline_bard_class() is cls
assert os.environ.get("SIMULATOR") == before
'''
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=60)


def test_offline_bard_batches_match_scalar_mechanics():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    calls = []
    backend = SimpleNamespace(model=None, data=None, ee_frame_id=0)
    def evaluate(*args):
        n = len(args[0])
        calls.append(n)
        # Vary every row so broadcasting one environment's mechanics cannot pass.
        mass = args[9][:, 0] + args[10][:, 0] + args[11][:, 0] + 1
        backend.pose = torch.eye(4).repeat(n, 1, 1)
        return SimpleNamespace(mass_matrix=mass[:, None, None]*torch.eye(8),
            bias=args[2].sum(1)[:, None].expand(n, 8).clone(),
            foot_jacobians=torch.zeros(n, 4, 6, 8),
            ee_jacobian=torch.zeros(n, 6, 8), base_jacobian=torch.zeros(n, 6, 8))
    backend.evaluate = evaluate
    backend.bard = SimpleNamespace(jacobian=lambda *args, **kwargs: (None, backend.pose))
    bard = object.__new__(a.OfflineBard)
    bard.backend, bard.device, bard.run = backend, "cpu", {"meta": {}}
    bard._finish_terms = lambda row, result, pose: result
    rows = [dict(base_position_t=np.zeros(3), base_quaternion_t=np.array([0,0,0,1]),
                 joint_position_t=np.full(2, i), base_velocity_world_t=np.zeros(3),
                 base_angular_velocity_world_t=np.zeros(3), joint_velocity_t=np.zeros(2),
                 mechanics_added_base_mass=np.array([i]), mechanics_base_com_bias=np.full(3, i/10),
                 mechanics_added_gripper_mass=np.array([i/2])) for i in range(5)]
    batched = bard.terms_batch(rows)
    assert calls == [5]
    scalar = [bard.terms(row) for row in rows]
    for left, right in zip(batched, scalar):
        for field in left:
            np.testing.assert_array_equal(left[field], right[field])
    assert bard.terms_batch([]) == []


def test_native_dynamics_variants_use_same_rows(tmp_path, monkeypatch):
    fixture(tmp_path/"raw", seeds=(0,))
    cfg = config(tmp_path)
    run = a.discover(cfg)[0][0]
    cfg.update(evaluation_seeds=[0], dynamics=True, inference_batch_size=3, dynamics_batch_size=2)
    constructions, batches = [], []
    class FakeBard:
        def __init__(self, run, *args, **kwargs):
            constructions.append(run["name"])
            self.owner = run["name"]
        def terms_batch(self, rows):
            batches.append(len(rows))
            return [self.terms(row) for row in rows]
        def terms(self, row):
            assert row["entry"]["source"] == self.owner
            assert "executed_torque_interval_mean" in row
            return dict(mass_matrix=np.eye(8), bias=np.zeros(8), foot_jacobians=np.zeros((4, 6, 8)),
                        ee_jacobian=np.eye(6, 8), base_jacobian=np.eye(6, 8), base_lever=np.zeros(3), ee_lever=np.zeros(3),
                        approximation=["synthetic mechanics"])
    monkeypatch.setattr(a, "OfflineBard", FakeBard)
    monkeypatch.setattr(a, "build_model", lambda *args: pytest.fail("Saved-prediction mode must not construct actors"))
    result = a._run_analysis(cfg)
    values = [r for r in result["table_summary"] if r["metric"] == "base_linear_rmse" and r["axis"] == "all"]
    assert {r["variant"] for r in values} == {"decoded", "measured_loading", "constant_velocity"}
    for variant in ("decoded", "measured_loading", "constant_velocity"):
        assert sum(r["contributing_samples"] for r in values if r["variant"] == variant) == result["selection"]["selected"]
    assert sum(r["transitions"] for r in result["dynamics_approximations"]) == result["selection"]["selected"]
    assert constructions == ["PILOT"]
    assert max(batches) == 2 and sum(batches) == result["selection"]["selected"]


def test_recorded_first_fallback_preserves_existing_fields(tmp_path,monkeypatch):
    fixture(tmp_path/"raw",seeds=(0,))
    cfg=config(tmp_path); cfg.update(evaluation_seeds=[0],recorded_native_only=False)
    before=a.run_analysis(cfg)
    remove_fields(tmp_path/"raw",["prediction_grf"])
    monkeypatch.setattr(a,"build_model",lambda *args: object())
    monkeypatch.setattr(a,"infer",lambda model,rows,device: dict(prediction_grf=np.ones((len(rows),12))*.001,
                                                             prediction_ee_force=np.ones((len(rows),3))*999))
    result=a.run_analysis(cfg)
    old=[r for r in before["table_summary"] if r["metric"]=="ee_force_rmse"]
    new=[r for r in result["table_summary"] if r["metric"]=="ee_force_rmse"]
    assert old==new
    assert result["prediction_sources"]["fallback:prediction_grf"]>0
    assert "fallback:prediction_ee_force" not in result["prediction_sources"]


def test_nonfinite_output_is_counted_not_zero_filled(tmp_path, monkeypatch):
    fixture(tmp_path/"raw", seeds=(0,))
    cfg = config(tmp_path)
    run = a.discover(cfg)[0][0]
    cfg.update(recording_roots=[str(run["path"])], evaluation_seeds=[0], inference_batch_size=2)
    original = a.physical_predictions
    def physical(pred, meta, row):
        values = original(pred, meta, row)
        if row["entry"]["step"] == 2:
            values["grf"][:] = np.nan
        return values
    monkeypatch.setattr(a, "physical_predictions", physical)
    result = a.run_analysis(cfg)
    issue = next(r for r in result["unavailable"] if r["component"] == "grf")
    assert issue["count"] == 2
    grf = next(r for r in result["table_summary"] if r["metric"] == "grf_rmse" and r["axis"] == "all")
    assert grf["contributing_samples"] == result["selection"]["selected"]-2
    assert grf["mean"] == pytest.approx(0.)  # Remaining finite predictions exactly match targets.

