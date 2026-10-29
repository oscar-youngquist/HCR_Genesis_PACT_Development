"""Offline-only fixtures; no simulator, model runner, or optimizer imports."""
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"notebooks"))
import exp3_film_oscillation as a
from test_exp1_analysis import metadata


def config(tmp):
    nb = json.loads((ROOT/"notebooks/exp3_film_oscillation_analysis.ipynb").read_text())
    text = "".join(nb["cells"][1]["source"])
    # Evaluate only the literal defaults, not notebook imports or execution.
    import ast
    assignment = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=="CONFIG" for t in n.targets))
    cfg = eval(compile(ast.Expression(assignment.value), "defaults", "eval"), {"ROOT": tmp})
    cfg.update(recording_roots=[str(tmp/"raw")], output_dir=str(tmp/"out"), evaluation_seeds=[0],
               loading=["loaded"], profiles=["pushing"], max_plot_windows=1, pca_samples=20,
               minimum_comparison_windows=2, max_windows=3)
    return cfg


def fixture(tmp):
    root = tmp/"raw"/"arbitrary_directory"
    root.mkdir(parents=True)
    m = metadata("b1z1_pact", 0)
    m.update(schema_version=1, experiment=2, dt=.02, prescribed_steps=500, checkpoint=str(tmp/"missing.pt"),
             checkpoint_sha256="abc", weights_unchanged=True, checkpoint_training_seed=123,
             dof_names=["leg", "z1_arm"], task_stages={"0": "nominal"},
             resolved_training={"policy": {"film_v02_enabled": False}})
    m["arguments"] = dict(profile="pushing", loading="loaded")
    m["evaluation"].update(move_start_s=1., velocity_ramp_s=.5, load_times_s=[3.,5.,7.,9.], stop_s=8.)
    m["resolved_environment"].update(env=dict(num_actions=2),
        normalization=dict(clip_actions=1., obs_scales=dict(lin_vel=2.,ang_vel=.25,ee_sphe_radius_cmd=1.,ee_sphe_pitch_cmd=1.,ee_sphe_yaw_cmd=1.)),
        control=dict(decimation=4,action_scale=.25,torque_scale=30.,torque_scale_overrides={"z1_arm":10.}),
        init_state=dict(default_joint_angles={"leg":0.,"z1_arm":0.}))
    t = np.arange(500)*.02
    x = np.sin(2*np.pi*2*t)
    pre = np.column_stack((np.ones(len(t)), np.cos(t), np.sin(t), np.cos(2*t)))
    gamma = np.column_stack((.1*x, .05*x, .01*x, .02*x))
    beta = .5*gamma
    ref = np.tile([.5,0,0], (len(t),1))
    measured = ref.copy(); measured[:,0] += .02*x
    condition = np.zeros((len(t),6)); condition[:,3] = -.022*x
    q = np.tile([0.,0.,0.,1.], (len(t),1))
    d = dict(film_condition_t=condition, film_features_pre_t=pre,
        film_features_post_t=(1+gamma)*pre+beta, film_gamma_t=gamma, film_beta_t=beta,
        trial_id=np.zeros(len(t),np.int64), step=np.arange(len(t)), time_seconds=t, successor_time_seconds=t+.02,
        valid=np.ones(len(t),bool), alive_for_metrics=np.ones(len(t),bool), physical_failure=np.zeros(len(t),bool), artificial_reset=np.zeros(len(t),bool),
        obs_t=np.column_stack((np.zeros((len(t),3)), ref)), ee_position_t_world=measured, ee_reference_t_world=ref,
        ee_reference_sphere_t=ref, base_quaternion_t=q, base_quaternion_xyzw=q,
        base_velocity_world_t=np.zeros_like(ref), base_angular_velocity_world_t=np.zeros_like(ref),
        ee_position_world=measured.copy(), ee_reference_world=ref, action_raw=np.zeros((len(t),4)))
    specs = []
    for i, start in enumerate(range(0,len(t),73)):
        name=f"transitions-{i}.npz"
        part={k:v[start:start+73] for k,v in d.items()}
        np.savez_compressed(root/name,**part)
        specs.append(dict(file=name,rows=len(part["step"])))
    (root/"metadata.json").write_text(json.dumps(m))
    (root/"records.json").write_text(json.dumps(dict(chunks=specs,fields={k:dict(shape=list(v.shape[1:]),dtype=str(v.dtype)) for k,v in d.items()})))
    (root/"trials.json").write_text(json.dumps([dict(trial_id=0)]))
    (root/"COMPLETE").write_text("complete")
    return m,d,root


def test_film_identity_and_temporal_decomposition():
    rng=np.random.default_rng(2)
    pre,gamma,beta=[rng.normal(size=(50,8)) for _ in range(3)]
    post=(1+gamma)*pre+beta
    delta,features,parameters=a.film_decomposition(pre,post,gamma,beta)
    np.testing.assert_allclose(features+parameters,np.diff(post,axis=0),atol=1e-12)
    np.testing.assert_allclose(delta,gamma*pre+beta,atol=1e-12)
    with pytest.raises(ValueError,match="reconstruction"):
        a.film_decomposition(pre,post+1,gamma,beta)
    np.testing.assert_array_equal(a.film_decomposition(pre,pre,np.zeros_like(pre),np.zeros_like(pre))[0],0)


def test_measured_condition_timing_and_frames(tmp_path):
    m,d,_=fixture(tmp_path)
    measured,cart=a.measured_condition(d,m)
    np.testing.assert_allclose(measured[:,3],-.02*np.sin(2*np.pi*2*d["time_seconds"]),atol=1e-12)
    np.testing.assert_allclose(cart[:,0],-measured[:,3])
    before=measured.copy()
    d["ee_position_world"] += 100  # Successor cannot affect actor-time reconstruction.
    np.testing.assert_array_equal(a.measured_condition(d,m)[0],before)
    from scipy.spatial.transform import Rotation
    q=Rotation.from_euler("xyz",[.2,-.3,.7]).as_quat()
    yaw=Rotation.from_quat(q).as_euler("xyz")[2]
    expected=Rotation.from_euler("z",-yaw).apply([1,2,3])
    np.testing.assert_allclose(a.inverse_rotate(np.array([[1.,2.,3.]]),q[None],True)[0],expected)


def test_linear_attribution_and_physical_scaling(tmp_path):
    m,d,_=fixture(tmp_path)
    rng=np.random.default_rng(3)
    heads={k:(rng.normal(size=(2,4)),rng.normal(size=2)) for k in ("position","torque")}
    pre,post=d["film_features_pre_t"],d["film_features_post_t"]
    _,_,p=a.film_decomposition(pre,post,d["film_gamma_t"],d["film_beta_t"])
    actions=np.concatenate([post@w.T+b for w,b in heads.values()],axis=1)
    attr,neutral,scales=a.attribute(pre,post,p,heads,actions,m)
    np.testing.assert_allclose(attr["position_film"],(actions[:,:2]-neutral[:,:2])*.25,atol=1e-12)
    np.testing.assert_allclose(attr["torque_film"],(actions[:,2:]-neutral[:,2:])*[30,10],atol=1e-12)
    with pytest.raises(ValueError,match="reconstruct"):
        a.attribute(pre,post,p,heads,actions+1,m)


def test_known_frequency_coherence_lag_and_short_window(tmp_path):
    cfg=config(tmp_path)
    t=np.arange(1000)*.02
    x=np.sin(2*np.pi*3.125*t)
    sp=a.spectrum(x,.02,cfg)
    assert sp["dominant"][0]==pytest.approx(3.125)
    assert sp["power"][0]==pytest.approx(.5,rel=.02)
    assert a.spectrum(x[:30],.02,cfg) is None
    rng=np.random.default_rng(42); x=rng.normal(size=1000); y=np.r_[np.zeros(5),x[:-5]]
    rel=a.relationship(x,y,.02,cfg)
    assert rel["lag_seconds"]==pytest.approx(.1)
    assert a.relationship(x,np.zeros_like(x),.02,cfg) is None
    assert np.isnan(a.spectrum(np.zeros(1000),.02,cfg)["dominant_mean"])


def test_nominal_torque_scaling_and_actual_clipping(tmp_path):
    m,d,_=fixture(tmp_path)
    n=len(d["step"])
    d.update(nominal_kp=np.full((n,2),100.), nominal_kd=np.full((n,2),2.),
        mechanics_kp_scale=np.full((n,2),2.), mechanics_kd_scale=np.full((n,2),3.),
        mechanics_motor_strength=np.full((n,2),.5), mechanics_feedback_tau_weight=np.full((n,1),.6),
        mechanics_feedforward_tau_weight=np.full((n,1),.8), joint_position_t=np.full((n,2),.1),
        joint_velocity_t=np.full((n,2),.2), actual_clipping_limits=np.full((n,2),5.))
    actual=np.tile([.5,.5,.1,.1],(n,1)); neutral=np.zeros_like(actual)
    scales=dict(position=np.array([.25,.25]),torque=np.array([30.,10.]))
    raw,clipped=a.nominal_torque_delta(actual,neutral,d,m,scales)
    original=.5*.6*(200*(-.1)-6*.2)
    requested=.5*(.6*(200*(.125-.1)-6*.2)+.8*np.array([3.,1.]))
    np.testing.assert_allclose(raw,np.broadcast_to(requested-original,raw.shape))
    np.testing.assert_allclose(clipped,np.broadcast_to(np.clip(requested,-5,5)-np.clip(original,-5,5),raw.shape))


def test_checkpoint_deserialization_is_frozen_and_simulator_free(tmp_path):
    torch=pytest.importorskip("torch")
    m,_,_=fixture(tmp_path); cfg=config(tmp_path)
    state={"position_head.weight":torch.arange(8,dtype=torch.float32).reshape(2,4),
           "position_head.bias":torch.zeros(2)}
    path=tmp_path/"weights.pt"
    torch.save({"model_state_dict":state},path)
    m.update(checkpoint=str(path),checkpoint_sha256=a.file_hash(path))
    rng=torch.random.get_rng_state().clone()
    heads,reason=a.load_heads(m,cfg)
    assert reason is None
    np.testing.assert_array_equal(heads["position"][0],state["position_head.weight"].numpy())
    assert torch.equal(torch.random.get_rng_state(),rng)
    assert "isaaclab" not in sys.modules and "legged_gym" not in sys.modules


def test_failed_head_parity_retains_feature_analysis(tmp_path):
    m,d,_=fixture(tmp_path); cfg=config(tmp_path)
    heads={k:(np.ones((2,4)),np.ones(2)) for k in ("position","torque")}
    metrics,traces,_,_,unavailable,_=a.analyze_window(d,m,heads,cfg)
    assert "relative_modulation_rms" in metrics and "position_film" not in traces
    assert any("max_abs_error" in s for s in unavailable)


def test_nonstationary_windows_do_not_report_coherence(tmp_path):
    m,d,_=fixture(tmp_path); cfg=config(tmp_path)
    cfg["stationary_window"]=False
    _,_,_,relations,unavailable,_=a.analyze_window(d,m,None,cfg)
    assert relations==[] and any("Nonstationary" in s for s in unavailable)


def test_gaps_failures_and_successor_alignment(tmp_path):
    _,d,_=fixture(tmp_path)
    rows=[{k:v[i] for k,v in d.items()} for i in (0,1,2,5,6,7)]
    rows[4]["physical_failure"]=True
    groups=a.contiguous_segments(rows,.02)
    assert [len(g) for g in groups]==[3,1]
    rows[0]["successor_time_seconds"]=99.
    with pytest.raises(ValueError,match="timestamps"):
        a.contiguous_segments(rows,.02)


def test_unavailable_checkpoint_architecture_and_aggregation(tmp_path):
    m,d,root=fixture(tmp_path); cfg=config(tmp_path)
    heads,reason=a.load_heads(m,cfg)
    assert heads is None and "unavailable" in reason.lower()
    m["resolved_training"]["policy"]["film_v02_enabled"]=True
    (root/"metadata.json").write_text(json.dumps(m))
    runs,excluded=a.discover(cfg)
    assert not runs and "V02" in excluded[0]["reason"]
    rows=[]
    for seed,tid,values in [(0,0,[0,0,0]),(0,1,[2]),(1,0,[3])]:
        rows += [dict(task="p",checkpoint="h",training_seed=1,profile="pushing",loading="loaded",kind="sustained",stage=1,metric="m",evaluation_seed=seed,trial_id=tid,value=v) for v in values]
    trials,seeds,summary=a.aggregate(pd.DataFrame(rows))
    assert seeds["mean"].tolist()==[1.,3.]
    assert summary.iloc[0]["mean"]==2.
    assert summary.iloc[0]["std"]==pytest.approx(np.sqrt(2))


def test_full_synthetic_exports(tmp_path):
    fixture(tmp_path); cfg=config(tmp_path)
    result=a.run_analysis(cfg)
    assert result["windows"] and result["summary"] and result["pca"]
    out=Path(cfg["output_dir"])
    assert (out/"COMPLETE").exists() and list(out.glob("*.pdf"))
    json.loads((out/"summary.json").read_text(),parse_constant=lambda _: pytest.fail("Nonstandard JSON"))
    assert "isaaclab" not in sys.modules and "legged_gym.envs" not in sys.modules


def test_notebook_smoke(tmp_path,monkeypatch):
    nbformat=pytest.importorskip("nbformat")
    from nbclient import NotebookClient
    fixture(tmp_path); cfg=config(tmp_path)
    monkeypatch.setenv("FILM_OSCILLATION_CONFIG",json.dumps(cfg))
    nb=nbformat.read(ROOT/"notebooks/exp3_film_oscillation_analysis.ipynb",as_version=4)
    NotebookClient(nb,timeout=180,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
    assert (Path(cfg["output_dir"])/"COMPLETE").exists()
