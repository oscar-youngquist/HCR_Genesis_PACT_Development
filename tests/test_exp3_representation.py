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
        ckpt.write_bytes(b"synthetic saved-prediction audit; not a Torch checkpoint")
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
    assert {r["benchmark"] for r in result["table_summary"]}=={"native_recorded", "shared"}
    assert not any(r["variant"]=="decoded" and r["metric"]=="base_linear_rmse" for r in result["table_summary"])
    assert (tmp_path/"output/shared_in_table.tex").exists()
    assert "NaN" not in (tmp_path/"output/summary.json").read_text()
    runs,_=a.discover(cfg);m=runs[0]["meta"];pos=copy.deepcopy(m)
    pos["resolved_environment"]["env"]["num_policy_actions"]=2
    assert a.input_contract(m)!=a.input_contract(pos)
    cfg["recorded_native_only"]=False
    assert any("disabled" in r["reason"] for r in result["unavailable"])


def test_common_transition_dispatch(tmp_path, monkeypatch):
    fixture(tmp_path/"raw", tasks=("b1z1_pact","b1z1_pact_ab4"), seeds=(0,))
    cfg=config(tmp_path)
    cfg.update(recorded_native_only=False, evaluation_seeds=[0], sampling_budget=24)
    cfg["methods"]["ab4"] = dict(task="b1z1_pact_ab4")
    # Independent deterministic toy heads test replay routing without Torch/Jupyter.
    def infer(model, data, device):
        n=len(data)
        return dict(prediction_z=np.ones((n,3)), prediction_ee_force=np.tile([.6,0,0],(n,1)),
                    prediction_base_wrench=np.zeros((n,6)), prediction_grf=np.ones((n,12))*.001)
    monkeypatch.setattr(a,"build_model",lambda *args: object())
    monkeypatch.setattr(a,"infer",infer)
    seen={}
    original=a.metric_rows
    def metrics(entry,method,benchmark,*args):
        if benchmark == "shared":
            seen.setdefault(method,set()).add(entry["transition_id"])
        return original(entry,method,benchmark,*args)
    monkeypatch.setattr(a,"metric_rows",metrics)
    result=a._run_analysis(cfg)
    assert seen["PILOT"]==seen["ab4"] and seen["PILOT"]
    assert any(r["benchmark"] == "shared" for r in result["table_summary"])


def test_torch_checkpoint_native_replay(tmp_path):
    torch=pytest.importorskip("torch")
    fixture(tmp_path/"raw",seeds=(0,)); runs,_=a.discover(config(tmp_path)); run=runs[0]
    from rsl_rl.modules.actor_critic_b1z1_pact import ActorCriticB1Z1PACT
    actor=ActorCriticB1Z1PACT(4,3,2,8,latent_dim=3,context_layers=[8],actor_layers=[8],critic_layers=[8],
        explicit_decoder_layers=[8],force_decoder_layers=[8],grf_decoder_layers=[8]).eval()
    torch.save(dict(model_state_dict=actor.state_dict()),run["checkpoint"])
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


def test_notebook_cpu_saved_prediction_smoke(tmp_path,monkeypatch):
    nbformat=pytest.importorskip("nbformat");nbclient=pytest.importorskip("nbclient")
    fixture(tmp_path/"raw", tasks=("b1z1_pact", "b1z1_ppo_pos"))
    cfg = config(tmp_path)
    cfg["methods"] = None
    from test_exp3_support import fixture as support_fixture
    support_fixture(tmp_path/"exp2")
    cfg["exp2_roots"] = [str(tmp_path/"exp2")]
    monkeypatch.setenv("EXP3_CONFIG",json.dumps(cfg))
    nb=nbformat.read(ROOT/"notebooks/exp3_representation_analysis.ipynb",as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell("assert 'torch' not in sys.modules and 'isaaclab' not in sys.modules"))
    nbclient.NotebookClient(nb,timeout=180,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
    nbformat.write(nb,tmp_path/"executed_exp3.ipynb")
    for name in ("transition_manifest.json","per_seed.csv","summary.csv","shared_in_table.tex","COMPLETE"):
        assert (tmp_path/"output"/name).stat().st_size>0


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

