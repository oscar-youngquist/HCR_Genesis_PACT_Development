"""Opt-in snapshots must not modify the native reference trajectory."""
import torch
import importlib.util
import json
from pathlib import Path
import numpy as np
from test_b1z1_directional_compliance import fixture


def test_raw_calibration_and_disabled_sign():
    m,k=fixture(3);m.cfg.ee_gain=[.005]*3;m.collect_diagnostics=True
    k['force'][:,2]=torch.tensor([30.,50.,-50.]);m.profile[:,4]=0
    m.advance(**k);d=m.diagnostics
    torch.testing.assert_close(d['raw_displacement'][:,2],torch.tensor([.15,.25,0.]))
    torch.testing.assert_close(d['selected_coefficients'][:,2],torch.tensor([1.,1.,0.]))
    assert all(not v.requires_grad for v in d.values())


def test_workspace_then_combined_geometry_and_filter():
    m,k=fixture(1);m.collect_diagnostics=True;m.cfg.ee_gain=[.005]*3
    k['nominal'][:,0]=.85;k['force'][:,0]=50
    k['target_safe']=lambda candidate,previous:previous+.5*(candidate-previous)
    r=m.advance(**k);d=m.diagnostics
    assert abs(float(d['raw_displacement'][0,0])-.25)<1e-6
    assert float(d['equilibrium_workspace'][0,0])<=.90001
    assert float(d['equilibrium_geometry'][0,0])<float(d['equilibrium_workspace'][0,0])
    assert float(d['filtered_candidate'][0,0])<float(d['equilibrium_geometry'][0,0])
    torch.testing.assert_close(d['accepted_position'],r.position)


def test_logging_parity_release_reset_and_snapshots():
    off,k=fixture(3);on,_=fixture(3);on.collect_diagnostics=True
    for i in range(80):
        k['force'][:,2]=30 if i<40 else 0
        a=off.advance(**k);b=on.advance(**k)
        for key in vars(a): torch.testing.assert_close(getattr(a,key),getattr(b,key),atol=0,rtol=0)
        assert off.diagnostics is None
    snapshot=on.diagnostics['accepted_position'].clone()
    on.reset(torch.tensor([1]));off.reset(torch.tensor([1]))
    torch.testing.assert_close(snapshot,on.diagnostics['accepted_position'])
    assert on.initialized.tolist()==[True,False,True]


def test_conditional_statistics_episode_maxima_and_legacy_unavailable(tmp_path):
    path=Path(__file__).resolve().parents[1]/'legged_gym/scripts/analyze_pilot_reference_stages.py'
    spec=importlib.util.spec_from_file_location('stage_analysis_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.measurements({},2,.02) is None
    episodes=[];folder=tmp_path/'seed_0'/'example';folder.mkdir(parents=True)
    for trial,values in enumerate(([.1,.2,.3],[.4])):
        n=len(values);z=np.zeros((n,3));raw=z.copy();raw[:,2]=values
        data={'reference_stage_'+k:v for k,v in dict(raw_displacement=raw,force_task=np.ones_like(z)*30,
            selected_coefficients=np.ones_like(z),nominal_position=z,equilibrium_workspace=raw,
            equilibrium_geometry=raw,filtered_candidate=raw*.5,accepted_position=raw*.5).items()}
        data.update(ee_task=raw*.5,alive_for_metrics=np.ones(n,bool),phase=np.ones(n)*2)
        if trial==0:data['alive_for_metrics'][-1]=False
        np.savez_compressed(folder/f'trial_{trial:04d}.npz',**data)
        episodes.append(dict(condition='example',seed=0,trial_id=trial,force_heading_direction=[0,0,1],failed=trial==0,pair_hash=str(trial)))
    (tmp_path/'metadata.json').write_text(json.dumps(dict(control_dt=.02,checkpoint_sha256='synthetic')))
    (tmp_path/'episodes.json').write_text(json.dumps(episodes));(tmp_path/'COMPLETE').touch()
    module.run(tmp_path)
    summary=json.loads((tmp_path/'reference_diagnostics.json').read_text())['summary']
    row=next(r for r in summary if r['mask']=='yielding_sustained' and r['metric']=='raw_norm')
    assert abs(row['mean']-.275)<1e-8
    assert abs(row['mean_episode_maximum']-.3)<1e-8
    assert abs(row['true_maximum']-.4)<1e-8
    assert row['samples']==3 and row['episodes']==2
    assert row['seed_sd'] is None
    assert module.describe([.15],.02)['fraction_gt_15cm']==0
    masks_data={k:np.repeat(v,3,axis=0) for k,v in data.items()}
    masks_data['reference_stage_force_task'][:,2]=[0,10,10]
    masks_data['reference_stage_selected_coefficients'][:,2]=[1,1e-9,0]
    _,masks,small=module.measurements(masks_data,2,.02)
    assert masks['yielding_active'].tolist()==[False,True,False]
    assert masks['protected_active'].tolist()==[False,False,True]
    assert small==1
