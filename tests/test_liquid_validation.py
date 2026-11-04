import numpy as np
import torch
from liquid_collection.validation.reconstruction import Reconstruction, tensor
from liquid_collection.validation.analyze import jvp_check, collocation_times
from liquid_collection.validation.run import payloads


def synthetic(dt=.005, duration=1.6):
    states=[]
    for t in np.arange(round(duration/dt)+1)*dt:
        theta=.7*t
        states.append(dict(time_s=t,x_O_W=np.array([t*t,0.,0.]),velocity_W=np.array([2*t,0,0,0,0,.7]),
            q_WB_wxyz=np.array([np.cos(theta/2),0,0,np.sin(theta/2)]),r_B=np.array([.1,.02,0.]),
            v_B=np.array([.03,0,0.]),S_B=np.array([0,0,.1]),mass=2.,gravity_W=np.array([0.,0.,-9.81]),
            interval_duration_s=dt,coupling_force_impulse_W=np.array([1.,2.,3.])*dt,
            coupling_moment_impulse_world_origin_W=np.array([4.,5.,6.])*dt))
    events=[dict(time_s=0.,held_acceleration_W=np.array([1.,0,0,0,0,0])),
            dict(time_s=.8,held_acceleration_W=np.array([4.,0,0,0,0,0]))]
    return states,events,states[1:]


def test_pose_derivative_frames_and_held_jump():
    c=Reconstruction(*synthetic())
    t=tensor([.3012,.7999,.8,.8001]);d=c.evaluate(t)
    np.testing.assert_allclose(d['u'][:,0],2*t,atol=1e-10)
    np.testing.assert_allclose(d['a'][:,0],2.,atol=1e-9)
    np.testing.assert_allclose(d['omega'][:,2],.7,atol=1e-10)
    np.testing.assert_allclose(d['alpha'],0.,atol=1e-7)
    worldF=(d['R']@d['F'][...,None])[...,0]
    worldM=(d['R']@d['tau'][...,None])[...,0]+torch.linalg.cross(d['x'],worldF)
    np.testing.assert_allclose(worldF,np.tile([1,2,3],(4,1)),atol=1e-10)
    np.testing.assert_allclose(worldM,np.tile([4,5,6],(4,1)),atol=1e-10)
    np.testing.assert_allclose(torch.linalg.vector_norm(d['z'][:,9:12],dim=-1),[1,1,4,4],atol=1e-10)
    # Independent v was not manufactured from r_dot; residual must remain nonzero.
    np.testing.assert_allclose(d['e_kin'],np.tile([-.03,0,0],(4,1)),atol=1e-10)


def test_jvp_and_duration_weighted_collocation():
    c=Reconstruction(*synthetic());n=Reconstruction(*synthetic(.00025))
    ts,measure=collocation_times(c,n,[.8],100,np.random.default_rng(2),.0001)
    assert len(ts)==100 and measure>0
    result=jvp_check(c,ts,[1e-5,2e-5,4e-5])
    assert result['autodiff_max']<1e-10
    assert max(result['finite_difference_max'].values())<1e-7


def test_interval_alignment():
    states,events,intervals=synthetic()
    for i,r in enumerate(intervals):
        r['coupling_force_impulse_W']=np.array([i+1.,0,0])*.005
    c=Reconstruction(states,events,intervals)
    d=c.evaluate(tensor([.004,.005,.005001,.009]))
    world=(d['R']@d['F'][...,None])[...,0]
    np.testing.assert_allclose(world[:,0],[1,1,2,2],atol=1e-10)


def test_payloads_and_shared_reset_settings():
    assert len(payloads())==3
    from types import SimpleNamespace
    from legged_gym.scripts.play_pact_water import apply_reset_termination,PLAY_RESET_TERMINATION
    cfg=SimpleNamespace(**{k:SimpleNamespace(**{x:'saved' for x in v}) for k,v in PLAY_RESET_TERMINATION.items()})
    apply_reset_termination(cfg)
    assert cfg.asset.terminate_after_contacts_on==['base']
    assert cfg.env.fail_to_terminal_time_s=='saved'


def test_native_replay_partial_reset_and_independent_audit(tmp_path):
    from test_liquid_collection import interval_row
    from liquid_collection.cadence import interval_fields
    from liquid_collection.storage import ChunkWriter
    from liquid_collection.validation.replay import replay,accounting,EVENTS
    from liquid_collection.audit import stream_rows
    native=tmp_path/'native';writer=ChunkWriter(native);sequence=0
    def emit(step,episode,duration,event=None):
        nonlocal sequence
        sequence+=1
        row=interval_row(step,.001)
        row['episode_id'][:]=episode
        row=interval_fields(row,duration,int(duration>0))
        for k in EVENTS: row[k]=torch.tensor([k==event])
        for k in ('r_B','v_B','S_B'): row[k]=torch.zeros(1,3,dtype=torch.float64)
        row.update(episode_native_step=torch.tensor([step]),episode_selected=torch.ones(1,dtype=torch.bool),
            held_valid=torch.ones(1,dtype=torch.bool),held_acceleration_W=torch.zeros(1,6),commands=torch.zeros(1,4))
        if not duration:
            from liquid_collection.cadence import IMPULSES
            for k in IMPULSES: row[k].zero_()
        row.update(force_interval_average_B=torch.zeros(1,3),moment_interval_average_O_B=torch.zeros(1,3))
        out={k:np.repeat(v.numpy(),2,axis=0) for k,v in row.items()}
        out['episode_id'][1]=0  # only environment zero resets at 3 ms
        if step>0:
            out['reset'][1]=False;out['terminal'][1]=False
        out.update(time_s=np.array(step*.001),native_step=np.array(step),record_sequence=np.array(sequence),target_grid_event=np.array(step%5==0))
        writer.append(out)
    emit(0,0,0,'reset')
    for step in range(1,4): emit(step,0,.001)
    emit(3,0,0,'terminal');emit(3,1,0,'reset')
    for step in range(4,8): emit(step,1,.001,'feature_update' if step==6 else None)
    writer.finish(dict(status='complete'))
    compact=tmp_path/'compact';replay(native,compact,.001)
    result=accounting(native,compact)
    assert result['status']=='pass',result
    partial=[r for r in stream_rows(compact/'boundaries') if r['interval_native_steps'].any()]
    assert [r['interval_native_steps'][0] for r in partial]==[3,2]
    assert [r['episode_id'].tolist() for r in partial]==[[0,0],[1,0]]


def test_static_newton_euler_and_original_integrated_loss():
    from liquid_collection.validation.analyze import integrated
    states,events,intervals=synthetic()
    for r in states:
        r['x_O_W'][:]=[1.,2.,3.];r['velocity_W'][:]=0;r['q_WB_wxyz'][:]=[1,0,0,0]
        r['v_B'][:]=0
        force=2*r['gravity_W']
        r['coupling_force_impulse_W']=force*.005
        r['coupling_moment_impulse_world_origin_W']=np.cross(r['x_O_W']+r['r_B'],force)*.005
    c=Reconstruction(states,events,intervals)
    d=c.evaluate(tensor([.101,.202]))
    for key in ('e_kin','e_Newton','e_Euler'): np.testing.assert_allclose(d[key],0,atol=1e-10)
    residuals=integrated(c,.3+np.arange(41)*.005)
    for value in residuals.values(): np.testing.assert_allclose(value,0,atol=1e-10)
    for r in intervals: r['coupling_force_impulse_W'][0]+=.005
    perturbed=Reconstruction(states,events,intervals)
    assert abs(integrated(perturbed,.3+np.arange(41)*.005)['P'][0]-.2)<1e-10


def test_collection_and_play_saved_settings_match():
    from pathlib import Path
    from copy import deepcopy
    from liquid_collection.adapters import restore_scope
    from liquid_collection.launch import PLATFORMS
    from legged_gym.scripts.play_pact_water import configure,apply_reset_termination,parser,PLAY_RESET_TERMINATION
    from legged_gym.scripts.pact_liquid_checkpoint import restore_inference_config,set_joint_dynamics_midpoints
    from legged_gym.envs.go1.go1_pact_water_eval.go1_pact_water_config import GO1PACTWaterCfg,GO1PACTWaterCfgPPO
    from legged_gym.envs.go2.go2_pact_water_eval.go2_pact_water_config import GO2PACTWaterCfg,GO2PACTWaterCfgPPO
    import json
    for platform,Cfg,Train in [('go1-dact',GO1PACTWaterCfg,GO1PACTWaterCfgPPO),('go2-dact',GO2PACTWaterCfg,GO2PACTWaterCfgPPO)]:
        manifest=PLATFORMS[platform];entry=json.loads(manifest.read_text())['policies'][0]
        checkpoint=(manifest.parent/entry['checkpoint']).resolve();source=(manifest.parent/entry['config']).resolve()
        play,collection=Cfg(),Cfg()
        for cfg in (play,collection): restore_inference_config(cfg,Train(),checkpoint,entry['robot'],source)
        configure(play,parser().parse_args(['--robot',entry['robot']]))
        restore_scope(collection,source)
        collection.liquid.joint_dynamics=set_joint_dynamics_midpoints(collection)
        apply_reset_termination(collection)
        assert collection.liquid.joint_dynamics==play.liquid.joint_dynamics
        for section,values in PLAY_RESET_TERMINATION.items():
            for key in values:
                assert getattr(getattr(collection,section),key)==getattr(getattr(play,section),key),(platform,section,key)


def test_native_kinematic_reference_is_explicit_interval_convention():
    from liquid_collection.validation.analyze import native_velocity_reference
    states,events,intervals=synthetic(.00025)
    for r in intervals: r['base_acceleration_interval_average_W']=np.array([2.,0,0,0,0,0])
    c=Reconstruction(states,events,intervals)
    times=np.array([.3012,.4121]);pose=c.evaluate(times)
    ref=native_velocity_reference(intervals,times,pose)
    np.testing.assert_allclose(ref['a'],[[2,0,0],[2,0,0]],atol=1e-12)
    np.testing.assert_allclose(ref['alpha'],0,atol=1e-12)
    np.testing.assert_allclose(ref['omega'][:,2],.7,atol=1e-12)
    np.testing.assert_allclose(ref['u'][:,0],2*np.ceil(times/.00025)*.00025,atol=1e-12)


def test_native_position_quantization_is_not_instantaneous_acceleration_truth():
    models=[]
    for dt in (.00025,.005):
        states,events,intervals=synthetic(dt)
        for row in states:
            row['x_O_W']=(row['x_O_W']+[10.,0.,0.]).astype(np.float32)
        models.append(Reconstruction(states,events,intervals))
    times=np.random.default_rng(12).uniform(.1,1.5,1000)
    errors=[float(torch.sqrt(torch.mean((model.evaluate(times)['a'][:,0]-2)**2))) for model in models]
    assert errors[0]>10*errors[1]
