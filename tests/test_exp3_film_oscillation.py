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
    cfg = a.default_config(eval(compile(ast.Expression(assignment.value), "defaults", "eval"), {"ROOT": tmp}))
    cfg['advanced_diagnostics']=True  # Retain coverage of the optional original diagnostics.
    cfg.update(pca=True,allow_unverified_attribution=True)
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


def test_recorded_associations_without_heads(tmp_path,monkeypatch):
    m,d,root=fixture(tmp_path); cfg=config(tmp_path)
    cfg.update(advanced_diagnostics=False,unmodulated_comparison=False,pca=False,max_windows=None,max_plot_windows=0,physical_comparison=False)
    monkeypatch.setattr(a,'load_heads',lambda *args: pytest.fail('Recorded analysis must not load a checkpoint'))
    t=d['time_seconds']; wave=np.sin(2*np.pi*2*t)
    d['film_features_pre_t'][:,0]=wave
    d['film_gamma_t'][:]=.5; d['film_beta_t'][:]=0
    d['film_features_post_t']=1.5*d['film_features_pre_t']
    d['action_raw'][:,0]=wave
    cfg['feature_bases']={m['checkpoint_sha256']:dict(center=[0]*4,axes=np.eye(4)[:2].tolist())}
    metrics,traces,_,relations,_,_=a.analyze_window(d,m,None,cfg)
    assert 'position_recorded' in traces and 'post_projection' in traces
    assert any(r['left']=='post_projection' and r['right']=='position_recorded' for r in relations)
    link=next(r for r in relations if r['left']=='position_recorded' and r['left_axis']==0 and r['right']=='ee_cartesian_error' and r['right_axis']==0)
    assert link['correlation']==pytest.approx(1.,abs=.01)
    result=a.run_analysis(cfg)
    assert result['windows'] and not result['evidence']
    assert result['config']['feature_bases'][m['checkpoint_sha256']]['samples']>0


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
    cfg["allow_unverified_attribution"]=False
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
    scheduled_fixture(tmp_path); cfg=config(tmp_path)
    cfg.update(max_windows=None, max_plot_windows=2, pca=False,advanced_diagnostics=False)
    monkeypatch.setenv("FILM_OSCILLATION_CONFIG",json.dumps(cfg))
    nb=nbformat.read(ROOT/"notebooks/exp3_film_oscillation_analysis.ipynb",as_version=4)
    NotebookClient(nb,timeout=180,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
    assert (Path(cfg["output_dir"])/"COMPLETE").exists()


def scheduled_fixture(tmp_path):
    """Recorded 50-Hz schedule, two interleaved environments, late failure in one."""
    m, old, root = fixture(tmp_path)
    m["prescribed_steps"] = 1400
    m["evaluation"].update(move_start_s=3., stop_s=20., velocity_ramp_s=.5,
                           load_times_s=[8.,13.,18.,23.], loads_n=[25.,55.,35.,0.])
    m["dof_names"] = ["FR_hip_joint", "z1_arm"]
    t = np.arange(1400)*.02
    d = {k: np.resize(v, (len(t), *v.shape[1:])) for k,v in old.items()}
    d.update(step=np.arange(len(t)), time_seconds=t, successor_time_seconds=t+.02)
    u = np.clip((t-8)/10,0,1); radius = .5+.1*(3*u*u-2*u*u*u)
    reference = np.column_stack((radius,np.zeros((len(t),2))))
    d["ee_reference_sphere_t"] = reference.copy()
    d["ee_reference_t_world"] = reference.copy(); d["ee_reference_world"] = reference.copy()
    d["obs_t"][:, -3:] = reference
    d["ee_position_t_world"] = reference+np.column_stack((.02*np.sin(2*np.pi*3*t),.01*np.sin(2*np.pi*4*t),np.zeros(len(t))))
    d["ee_position_world"] = d["ee_position_t_world"].copy()
    d["executed_torque_interval_mean"] = np.column_stack((np.sin(2*np.pi*3*t),2*np.sin(2*np.pi*3*t)))
    load = np.select([t<8,t<13,t<18,t<23],[0,25,55,35],default=0)
    d["ee_force_applied_world_interval_mean"] = np.column_stack((-load,np.zeros((len(t),2))))
    # Two trials interleaved like a vectorized recording, never concatenated for spectra.
    combined = {k:np.repeat(v,2,axis=0) for k,v in d.items()}
    combined["trial_id"] = np.tile([0,1],len(t))
    failed = (combined["trial_id"]==1) & (combined["time_seconds"]>=22)
    combined["physical_failure"][failed] = True
    specs=[]
    for i,start in enumerate(range(0,len(t)*2,197)):
        part={k:v[start:start+197] for k,v in combined.items()}
        name=f"scheduled-{i}.npz"; np.savez_compressed(root/name,**part)
        specs.append(dict(file=name,rows=len(part["step"])))
    (root/"metadata.json").write_text(json.dumps(m))
    (root/"records.json").write_text(json.dumps(dict(chunks=specs,fields={k:dict(shape=list(v.shape[1:]),dtype=str(v.dtype)) for k,v in combined.items()})))
    (root/"trials.json").write_text(json.dumps([dict(trial_id=0),dict(trial_id=1)]))
    return m,d,root


def test_adaptive_loaded_windows_and_coverage(tmp_path):
    m,d,_=scheduled_fixture(tmp_path); cfg=config(tmp_path)
    cfg.update(max_windows=None,max_plot_windows=2,pca=False)
    plan=a.spectral_plan(99,.02,cfg)
    assert plan["available"] and plan["segments"]>=4 and plan["nperseg"]<64
    assert plan["usable_low_hz"]>=plan["resolution_hz"]
    assert not a.spectral_plan(99,.02,{**cfg,"segment_length_policy":"fixed"})["available"]
    assert a.spectrum(np.ones(30),.02,cfg) is None
    assert a.spectrum(np.ones(99),.02,{**cfg,"frequency_band_hz":[.1,.2]}) is None
    result=a.run_analysis(cfg)
    loaded=[w for w in result["windows"] if w["kind"]=="sustained" and w["scheduled_load_n"]>0]
    assert {w["load_stage"] for w in loaded}=={0,1,2}
    assert any(w["temporal_available"] and w["analysis_mode"]=="stationary" for w in loaded)
    assert any(w["temporal_available"] and w["analysis_mode"]=="moving_reference_detrended" for w in loaded)
    assert all(not (w["actual_start"]<event<w["actual_end"]) for w in loaded for event in m["evaluation"]["load_times_s"])
    assert all(w["actual_end"]<=22 for w in result["windows"] if w["trial_id"]==1)
    assert (Path(cfg["output_dir"])/"coverage.csv").exists()
    assert len(result["plot_selection"])==2
    assert len({w["trial_id"] for w in result["plot_selection"]})==2


def test_attributed_temporal_links_directions_and_groups(tmp_path):
    m,d,_=scheduled_fixture(tmp_path); cfg=config(tmp_path)
    mask=(d["time_seconds"]>=20.75)&(d["time_seconds"]<22.75)
    d={k:v[mask] for k,v in d.items()}
    heads={name:(np.array([[1.,.1,.1,.1],[.1,1.,.1,.1]]),np.zeros(2)) for name in ("position","torque")}
    d["action_raw"]=np.concatenate([d["film_features_post_t"]@w.T+b for w,b in heads.values()],1)
    metrics,traces,spectra,relations,_,units=a.analyze_window(d,m,heads,cfg)
    assert a.joint_groups(m,cfg)=={"arm":[1],"leg":[0]}
    np.testing.assert_allclose(traces["ee_load_parallel_error"][:,0],-(d["ee_position_t_world"]-d["ee_reference_t_world"])[1:,0])
    for name in ("position_film","position_parameter","torque_film","torque_parameter"):
        assert name+"_arm" in traces and name+"_leg" in traces
        assert any(r["right"]==name and r["left"]=="ee_cartesian_error" for r in relations)
        assert any(r["left"]==name and r["right"]=="executed_torque_interval_mean" for r in relations)
        assert any(r["left"]==name and r["right"]=="successor_ee_error" for r in relations)
    assert "ee_cartesian_error/component_x/signed_bias" in metrics
    assert "position_film_arm/component_z1_arm/band_power" in metrics
    assert spectra["position_film_arm"]["psd"].shape[1]==1
    assert units["position_film_arm"]=="rad"
    assert not any(r["left"]=="parameter_change" for r in relations)
    assert any(r["left"]=="position_parameter" and r["right"]=="position_film" for r in relations)
    d["ee_force_applied_world_interval_mean"][:]=0
    assert a.load_projections(d,cfg) is None


def test_selection_is_deterministic_and_covers_categories():
    rows=[]
    for profile in ("pushing","pulling"):
        for loading in ("loaded","unloaded"):
            for tid in range(4):
                rows.append(dict(profile=profile,loading=loading,kind="sustained" if tid%2 else "load_response",
                    checkpoint="a" if tid<2 else "b",evaluation_seed=tid%2,stage=tid%2,
                    oscillation_severity=tid*.01,run=f"{profile}/{loading}",trial_id=tid,actual_start=1.,window_id=len(rows)))
    chosen,omitted=a.select_plot_windows(rows,8)
    again,_=a.select_plot_windows(list(reversed(rows)),8)
    assert [w["window_id"] for w in chosen]==[w["window_id"] for w in again]
    for field in ("profile","loading","checkpoint","evaluation_seed"):
        assert {w[field] for w in chosen}=={w[field] for w in rows}
    assert len({(w["run"],w["trial_id"]) for w in chosen})==8


def test_precision_explanation_is_not_looser_tolerance():
    rng=np.random.default_rng(9)
    x=rng.normal(size=(100,128)).astype('float32'); w=rng.normal(size=(5,128)).astype('float32'); b=np.zeros(5)
    saved=a.reduced_precision_operand(x)@a.reduced_precision_operand(w).T+b
    audit=a.head_parity(x,w,b,saved,3e-5,3e-4)
    assert audit['status']=='approximate_precision_explained'
    assert max(audit['precision_residual_max'])<1e-10
    assert np.all(np.array(audit['max_abs_error'])<=audit['error_bound_max'])
    assert a.head_parity(x,w,b,saved+.1,3e-5,3e-4)['status']=='unavailable'
    exact=x.astype(float)@w.astype(float).T
    assert a.head_parity(x,w,b,exact,3e-5,3e-4)['status']=='strict'


def test_fixed_varying_modulation_and_permissive_status(tmp_path):
    m,d,_=fixture(tmp_path); cfg=config(tmp_path)
    pre=d['film_features_pre_t']; gamma=d['film_gamma_t']; beta=d['film_beta_t']
    post=(1+gamma)*pre+beta; fixed=(1+gamma.mean(0))*pre+beta.mean(0)
    heads={k:(np.ones((2,4)),np.zeros(2)) for k in ('position','torque')}
    _,_,param=a.film_decomposition(pre,post,gamma,beta)
    actions=np.concatenate([post@w.T+b for w,b in heads.values()],1)
    out,_,_=a.attribute(pre,post,param,heads,actions,m,fixed=fixed)
    for head in heads:
        np.testing.assert_allclose(out[head+'_persistent_effect']+out[head+'_varying_effect'],out[head+'_film'],atol=1e-12)
    constant=np.tile(gamma.mean(0),(len(pre),1)); bconst=np.tile(beta.mean(0),(len(pre),1))
    post=(1+constant)*pre+bconst
    actions=np.concatenate([post@w.T+b for w,b in heads.values()],1)
    out,_,_=a.attribute(pre,post,np.zeros_like(pre[:-1]),heads,actions,m,fixed=post)
    assert not out['position_varying_effect'].any()
    metrics,traces,_,_,notes,units=a.analyze_window(d,m,heads,cfg)
    assert 'position_film' in traces and any('UNVERIFIED' in note for note in notes)
    assert units['head_parity']['position']['status']=='unverified_algebraic_only'


def test_base_arm_velocity_frames_and_time(tmp_path):
    from exp3_oscillation_physical import motion_signals
    m,d,_=fixture(tmp_path)
    d['base_position_t']=np.zeros_like(d['ee_position_t_world'])
    d['base_velocity_world_t'][:]=[1,0,0]; d['base_angular_velocity_world_t'][:]=[0,0,2]
    rotation=np.cross(d['base_angular_velocity_world_t'],d['ee_position_t_world'])
    relative=np.tile([0,0,.3],(len(rotation),1))
    d['ee_velocity_t_world']=d['base_velocity_world_t']+rotation+relative
    traces,reason=motion_signals(d); assert reason is None
    np.testing.assert_allclose(traces['arm_relative'],relative)
    np.testing.assert_allclose(traces['base_translation']+traces['base_rotation']+traces['arm_relative'],traces['ee_motion'])
    d['ee_velocity_world']=np.full_like(relative,999.)  # Successor must not contaminate actor-time identity.
    np.testing.assert_allclose(motion_signals(d)[0]['arm_relative'],relative)
    d['base_quaternion_t'][:]=[0,0,np.sin(np.pi/4),np.cos(np.pi/4)]
    np.testing.assert_allclose(motion_signals(d)[0]['base_translation'],np.tile([0,-1,0],(len(relative),1)),atol=1e-12)


def test_matched_physical_methods_and_frequency_contract(tmp_path):
    import shutil
    import exp3_oscillation_physical as physical
    m,d,root=scheduled_fixture(tmp_path)
    for task in ('b1z1_unifp_original','b1z1_ppo_pos'):
        dest=root.parent/task; shutil.copytree(root,dest)
        meta=copy.deepcopy(m); meta['task']=task; meta['checkpoint_sha256']=task
        meta['resolved_environment']['control']['stiffness']={'hip':250.}
        (dest/'metadata.json').write_text(json.dumps(meta))
    cfg=config(tmp_path); cfg.update(comparison_plot_windows=0,comparison_max_windows=None)
    result=physical.analyze(cfg,tmp_path/'physical')
    assert {r['task'] for r in result['matched']}==set(cfg['comparison_tasks'])
    assert len({(w['spectral_plan']['nperseg'],w['spectral_plan']['resolution_hz'],w['spectral_plan']['usable_low_hz'],w['spectral_plan']['usable_high_hz']) for w in result['windows']})==1
    assert any('No matched unloaded' in s for s in result['diagnostics'])
    # Different physical protocols never enter the same matched comparison.
    path=root.parent/'b1z1_ppo_pos'/'metadata.json'; altered=json.loads(path.read_text())
    altered['evaluation']['loads_n'][0]=40; path.write_text(json.dumps(altered))
    other=physical.analyze(cfg,tmp_path/'incompatible')
    assert not other['matched'] and other['summary']


def test_reference_amplitudes_and_phase_guard(tmp_path):
    m,d,_=fixture(tmp_path); cfg=config(tmp_path)
    d['film_gamma_t'][:]=1.; d['film_beta_t'][:]=0.
    d['film_features_post_t']=2*d['film_features_pre_t']
    heads={name:(np.ones((2,4)),np.zeros(2)) for name in ('position','torque')}
    d['action_raw']=np.concatenate([d['film_features_post_t']@w.T for w,b in heads.values()],1)
    cfg['coherence_threshold']=1.01  # Deliberately suppress all phase interpretations.
    metrics,_,_,_,_,_=a.analyze_window(d,m,heads,cfg)
    assert metrics['position_arm/actual_minus_fixed_rms']==pytest.approx(0.)
    assert metrics['position_arm/fixed_minus_unmodulated_rms']>0
    phases=[v for k,v in metrics.items() if '/phase_at_ee_' in k]
    assert phases and np.isnan(phases).all()


def test_high_low_trial_uncertainty_and_mode_isolation(tmp_path):
    rows=[]
    for quality in ('strict','unverified'):
        for mode in ('stationary','moving_reference_residual'):
            for trial in range(3):
                for step in range(4):
                    identity=dict(checkpoint='a',evaluation_seed=0,trial_id=trial,profile='pushing',loading='loaded',kind='sustained',stage=1,
                                  window_id=len(rows),analysis_mode=mode,attribution_quality=quality)
                    for metric,value in [('ee_cartesian_error/oscillatory_rms',step),('ee_cartesian_error/rms',.1),('relative_modulation_rms',trial+step)]:
                        rows.append(dict(**identity,metric=metric,value=value))
    a.compare_windows(pd.DataFrame(rows),tmp_path,dict(minimum_comparison_windows=2,error_match_bin_m=.02))
    result=pd.read_csv(tmp_path/'high_low_windows.csv')
    assert set(result.attribution_quality)=={'strict','unverified'}
    assert set(result.analysis_mode)=={'stationary','moving_reference_residual'}
    assert (result.trials==3).all() and (result.seeds==1).all()
    assert result.sd.isna().all()
    assert (result[result.metric=='relative_modulation_rms'].trial_sd>0).all()


def test_unverified_references_excluded_from_evidence():
    rows=[dict(metric='position_arm/fixed_minus_unmodulated_rms',kind='sustained',mean=100.,attribution_quality='unverified')]
    report=a.explanatory_report(rows,{},[])
    evidence=next(r for r in report if r['candidate']=='Persistent FiLM scaling')
    assert 'unverified values excluded' in evidence['supporting']
