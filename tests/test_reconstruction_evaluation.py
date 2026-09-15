"""Focused CPU checks; simulator integration is optional and explicitly separate."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import numpy as np
import pandas as pd
import pytest

from reconstruction_eval.collect import export_metadata
from reconstruction_eval.schema import OBS_FIELDS, EXPLICIT_FIELDS, transform, reorder, layout, validate
from reconstruction_eval.io import ChunkWriter, read_dataset
from reconstruction_eval.analysis import contact_phases, phase_and_window, analyze, summarize, GROUPS, legacy_availability
from reconstruction_eval.dynamics import rotation_xyzw, momentum_error
from reconstruction_eval.probes import scenario_split, ridge_probe
from reconstruction_eval.snapshot import EvaluationHook, base_velocity_command

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def metadata(tmp_path):
    export_metadata('go1_pact', tmp_path/'meta.json')
    return json.loads((tmp_path/'meta.json').read_text())


def test_schema_roundtrip_ordering_and_reject_legacy(metadata):
    validate(metadata)
    assert sum(w for _,w in metadata['decoder_fields']) == 276
    values = np.random.default_rng(0).normal(size=(4,20,57))
    normalized = transform(values, OBS_FIELDS, metadata)
    np.testing.assert_allclose(transform(normalized, OBS_FIELDS, metadata, inverse=True), values, atol=1e-12)
    other = copy.deepcopy(metadata)
    other['joint_names'] = metadata['joint_names'][::-1]
    other['foot_names'] = metadata['foot_names'][::-1]
    fields = metadata['privileged_fields']
    x = np.arange(sum(w for _,w in fields))[None,:].astype(float)
    reordered = reorder(x, fields, metadata, other)
    np.testing.assert_equal(reorder(reordered, fields, other, metadata), x)
    grf = layout(fields)['grf']
    np.testing.assert_equal(reordered[:,grf].reshape(1,4,3), x[:,grf].reshape(1,4,3)[:,::-1])
    other['decoder_fields'] = other['privileged_fields']
    with pytest.raises(ValueError, match='exclude exactly'):
        validate(other)
    with pytest.raises(ValueError, match='no truncation'):
        transform(np.zeros((1,56)), OBS_FIELDS, metadata)


def test_contact_transitions_and_gaps():
    force = np.zeros((9,4,3))
    force[2:6,:,2] = 100
    phases = contact_phases(force, np.arange(9)*.01, np.arange(9), threshold=10, transition_window=.01)
    assert phases[:,0].tolist() == ['unknown','swing','touchdown','touchdown','steady_stance',
                                     'steady_stance','liftoff','liftoff','swing']
    steps = np.arange(9); steps[4:] += 2
    assert contact_phases(force, np.arange(9)*.01, steps)[4,0] == 'unknown'


def test_phases_do_not_cross_interleaved_environment_or_episode():
    data = dict(environment=np.array([0,1,0,1,0,0]), episode=np.array([0,0,0,0,1,1]),
                step=np.array([0,0,1,1,0,1]), timestamp=np.array([0,0,.01,.01,0,.01]),
                timestamp_end=np.array([.01,.01,.02,.02,.01,.02]),
                measured_grf=np.zeros((6,4,3)), injection=np.array([0,1,0,0,0,0],bool),
                unaccounted_injection=np.zeros(6,bool))
    data['measured_grf'][[2,4,5],:,2] = 30
    phases, windows = phase_and_window(data, 1., .03, .5)
    assert phases[2,0] == 'touchdown'
    assert phases[4,0] == 'unknown'
    assert phases[5,0] == 'steady_stance'
    assert windows.tolist() == ['ordinary','impulse_event','ordinary','recovery','ordinary','ordinary']


def test_momentum_balance_gravity_and_velocity_jump():
    mass, dt = 10., .1
    gravity = np.array([0.,0.,-9.81*mass])
    support = -gravity
    linear, angular = momentum_error(np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),
                                     (support+gravity)*dt,np.zeros(3))
    np.testing.assert_equal(linear,0)
    np.testing.assert_equal(angular,0)
    jump = np.array([0.,mass*2,0.])
    linear, _ = momentum_error(np.zeros(3),jump,np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),linear_jump=jump)
    np.testing.assert_equal(linear,0)
    # Ballistic whole-body momentum must include gravity.
    linear, _ = momentum_error(np.zeros(3),gravity*dt,np.zeros(3),np.zeros(3),gravity*dt,np.zeros(3))
    np.testing.assert_equal(linear,0)
    R = rotation_xyzw([0,0,np.sqrt(.5),np.sqrt(.5)])
    np.testing.assert_allclose(R@np.array([1.,0.,0.]),[0.,1.,0.],atol=1e-12)
    origin=np.array([100.,-70.,.5])
    # Shifting scene origin must not contaminate the angular residual with the
    # otherwise independent linear force-quadrature error.
    dp=np.array([.1,.2,-.1])
    linear,angular=momentum_error(np.zeros(3),dp,np.zeros(3),np.cross(origin,dp),
                                 np.zeros(3),np.zeros(3),window_origin=origin)
    np.testing.assert_allclose(angular,0.,atol=1e-12)


def test_world_velocity_injection_converts_genesis_angular_dofs():
    q=np.zeros((1,19));q[0,3:7]=[0.,0.,np.sqrt(.5),np.sqrt(.5)]
    world=np.array([[1.,2.,3.,.4,.5,.6]])
    command=base_velocity_command(q,world)
    np.testing.assert_allclose(command[:,:3],world[:,:3])
    np.testing.assert_allclose(rotation_xyzw(q[0,3:7])@command[0,3:],world[0,3:])
    assert not np.allclose(command[0,3:],world[0,3:])


def test_rmse_pools_squared_errors_and_bootstraps_episodes():
    identity = {k:'x' for k in GROUPS}
    records = [dict(identity,episode_key='a',absolute_error_sum=1.,squared_error_sum=1.,scalar_count=1,valid_sample_count=1),
               dict(identity,episode_key='b',absolute_error_sum=9.,squared_error_sum=27.,scalar_count=3,valid_sample_count=3)]
    result = summarize(pd.DataFrame(records),100,1).iloc[0]
    assert result.rmse == pytest.approx(np.sqrt(7.))
    assert result.mae == 2.5 and result.valid_episode_count == 2
    assert result.rmse_ci_low <= result.rmse <= result.rmse_ci_high


def dataset(tmp_path, metadata, source, n=8):
    """Two failed episodes, pre-reset targets intact; paired predictions."""
    fields = metadata['privileged_fields']
    raw = np.zeros((n, sum(w for _,w in fields)))
    raw[:,layout(fields)['payload']] = 4.
    explicit = np.zeros((n,16)); explicit[:,layout(EXPLICIT_FIELDS)['payload']] = 4.
    select = [i for name,sl in layout(fields).items() if name != 'grf' for i in range(sl.start,sl.stop)]
    pred = transform(raw[:,select], metadata['decoder_fields'], metadata, clip=True)
    data = dict(target=raw, explicit_current=explicit, explicit_next=explicit+1., payload=np.full((n,1),4.),
                com=np.zeros((n,3)), environment=np.zeros(n,int), episode=np.repeat([0,1],n//2),
                step=np.tile(np.arange(n//2),2), timestamp=np.tile(np.arange(n//2)*.01,2),
                timestamp_end=np.tile((np.arange(n//2)+1)*.01,2),
                terminal=np.tile([False,False,False,True],2), failure=np.ones(n,bool),
                reset_crossing=np.zeros(n,bool), unaccounted_injection=np.zeros(n,bool),
                injection=np.zeros(n,bool), history_valid=np.ones(n,bool), measured_grf=np.zeros((n,4,3)),
                scenario=np.repeat(['1:plane:nominal:0:0','1:plane:nominal:0:1'],4))
    models = []
    for method in ['pact','abl3']:
        models.append(dict(method=method, checkpoint_sha256=method, training_seed=None,metadata=metadata))
        data[method+'__decoder'] = pred.copy()
        data[method+'__explicit'] = transform(explicit, EXPLICIT_FIELDS, metadata)
        data[method+'__grf'] = np.zeros((n,12))
        data[method+'__latent'] = np.zeros((n,16))
    # One method has a bad GRF prediction. Both use the same valid GRF rows.
    data['abl3__grf'][0] = np.nan
    manifest = dict(models=models,source_metadata=metadata,rollout_source=source,rollout_checkpoint=source,
                    condition='nominal',terrain='plane',scenario_seed=1)
    directory=tmp_path/source
    writer = ChunkWriter(directory,manifest,4)
    writer.append({k:v[:4] for k,v in data.items()})
    writer.append({k:v[4:] for k,v in data.items()})
    writer.close()
    return directory


def test_chunked_analysis_failed_episodes_alignment_and_matched_masks(tmp_path, metadata):
    from reconstruction_eval.reporting import discover_datasets, dataset_inventory, comparison_table, plot_metric
    pact, abl = [dataset(tmp_path,metadata,source) for source in ['pact','abl3']]
    old = tmp_path/'go1_abl3_old.csv'
    old.write_text('legacy,data\n')
    result = analyze([pact,abl,old],tmp_path/'analysis',controlled=True,bootstrap=10)
    assert set(discover_datasets(tmp_path)) == {pact,abl}
    inventory = dataset_inventory([pact,abl])
    assert set(inventory.method) == {'pact','abl3'}
    assert set(inventory.rollout_source) == {'pact','abl3'}
    table = comparison_table(result, ['payload'])
    assert set(table.index.get_level_values('rollout_source')) == {'pact','abl3'}
    assert set(table.columns.get_level_values('method')) == {'pact','abl3'}
    with pytest.raises(ValueError):
        comparison_table(pd.concat([result,result]), ['payload'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plot_metric(result, 'payload')
    assert len(fig.axes) == 2  # separate rollout-source panels
    plt.close(fig)
    assert len(pd.read_csv(tmp_path/'analysis/unavailable.csv').query("metric == 'grf'")) == 1
    current = result.query("metric == 'explicit_current_payload' and window == 'all'")
    nxt = result.query("metric == 'explicit_next_payload' and window == 'all'")
    assert (current.mae == 0).all()
    assert (nxt.mae == 1).all()
    grf = result.query("metric == 'grf' and phase == 'all' and window == 'all'")
    assert set(grf.valid_sample_count) == {7}
    assert set(grf.valid_episode_count) == {2}
    assert not any(c.startswith('Unnamed') for c in pd.read_csv(tmp_path/'analysis/summary.csv').columns)
    assert not any(c.startswith('Unnamed') for c in pd.read_csv(tmp_path/'analysis/comparison.csv').columns)
    for report in ('summary.csv', 'episode_errors.csv', 'inventory.csv', 'comparison.csv'):
        assert not any('checkpoint' in c for c in pd.read_csv(tmp_path/'analysis'/report).columns)
    with pytest.raises(ValueError,match='Unbalanced'):
        analyze([pact],tmp_path/'bad',controlled=True)
    # New anonymous session metadata retains source separation without logging
    # model paths/hashes. Legacy identities above remain readable internally.
    for directory in (pact, abl):
        manifest_path = directory/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.pop('rollout_checkpoint')
        for model in manifest['models']:
            model.pop('checkpoint_sha256')
            model.update(comparison_id='shared-session', on_policy=model['method']==manifest['rollout_source'])
        manifest_path.write_text(json.dumps(manifest))
    anonymous = analyze([pact,abl],tmp_path/'anonymous',controlled=True)
    assert set(anonymous.comparison) == {'common_histories','on_policy'}
    (pact/'chunk_00000.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        read_dataset(pact)


def test_grouped_probe_train_only_normalization():
    groups = np.repeat(np.arange(20).astype(str),5)
    splits, mapping = scenario_split(groups,3)
    for g in set(groups):
        assert len(set(splits[groups==g])) == 1
    x = np.random.default_rng(2).normal(size=(100,3))
    y = np.c_[x[:,0]*2,x[:,1],x[:,2],x[:,0]+x[:,1]]
    e, baseline, params = ridge_probe(x,y,splits)
    assert np.square(e).mean() < np.square(baseline).mean()*.01
    x[splits=='test'] += 1000
    _,_,new_params = ridge_probe(x,y,splits)
    assert params['feature_mean'] == new_params['feature_mean']
    assert params['feature_scale'] == new_params['feature_scale']
    assert params['selected_alpha'] == new_params['selected_alpha']


def test_probe_pipeline_groups_shared_families_across_terrains(tmp_path):
    from reconstruction_eval.probes import run_probes
    datasets=[]
    for source in ['pact','abl3']:
        for terrain in ['plane','rough']:
            n=24
            episode=np.repeat(np.arange(6),4)
            x=(episode+1).astype(float)
            data=dict(history_valid=np.ones(n,bool),reset_crossing=np.zeros(n,bool),
                unaccounted_injection=np.zeros(n,bool),terminal=np.tile([0,0,0,1],6).astype(bool),
                environment=np.zeros(n,int),episode=episode,step=np.tile(np.arange(4),6),
                payload=x[:,None],com=np.c_[x*.01,x*.02,x*.03],
                scenario=np.asarray([f'42:{terrain}:payload:0:{e}' for e in episode]),
                pact__latent=np.c_[x,x*x],abl3__latent=np.c_[x*2,x*x*3])
            meta=dict(rollout_source=source,condition='payload',terrain=terrain,
                      models=[dict(method=m,checkpoint_sha256=m) for m in ['pact','abl3']])
            datasets.append(('',meta,data))
    result=run_probes(datasets,tmp_path,1)
    manifest=json.loads((tmp_path/'probe_manifest.json').read_text())
    assert len(manifest['scenario_split']) == 6  # not 12 terrains or 24 sources
    assert set(result.rollout_source) == {'pact','abl3'}
    assert set(result.terrain) == {'plane','rough'}
    assert set(result.valid_episode_count) == {2}


def test_per_episode_mass_com_readback_and_independent_sampling(monkeypatch):
    # Minimal tensor interface exercises the actual reset helper with a fake
    # simulator setter/readback, not a second implementation of its sampler.
    monkeypatch.setitem(sys.modules,'torch',NS(float32=np.float32,
        as_tensor=lambda value, **kwargs: np.asarray(value,dtype=kwargs.get('dtype'))))
    mass, com = np.zeros((2,1)), np.zeros((2,1,3))
    def set_mass(v, link, ids): mass[ids] = v
    def set_com(v, link, ids): com[ids] = v
    robot = NS(link_start=0,set_mass_shift=set_mass,set_COM_shift=set_com,
        _solver=NS(get_links_mass_shift=lambda link,ids:mass[ids], get_links_COM_shift=lambda link,ids:com[ids]))
    sim=NS(_device='cpu',_base_link_index=0,_robot=robot,_added_base_mass=np.zeros((2,1)),_base_com_bias=np.zeros((2,3)))
    hook=EvaluationHook(NS(num_envs=2),17,'payload',[2.,8.],[-.2,.2,-.1,.1,0.,.3])
    hook.reset(sim,np.array([0,1]))
    first_mass,first_com=mass.copy(),com.copy()
    hook.reset(sim,np.array([0]))
    assert mass[1] == first_mass[1]
    np.testing.assert_equal(com[1],first_com[1])
    assert mass[0] != first_mass[0]
    assert not np.allclose(com[0],first_com[0])
    robot._solver.get_links_mass_shift=lambda link,ids:np.zeros((len(ids),1))
    with pytest.raises(RuntimeError,match='did not apply'):
        hook.reset(sim,np.array([0]))


def test_opt_in_hooks_capture_before_reset_in_both_tasks():
    for task in ['pact','abl3']:
        path=ROOT/f'legged_gym/envs/go1/go1_{task}/go1_{task}.py'
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='post_physics_step')
        code=ast.unparse(method)
        assert code.index('check_termination()') < code.index('reconstruction_evaluation.capture(self)') < code.index('self.reset_idx(env_ids)')
    # Every old file is explicitly unavailable; no inferred target slices.
    assert set(legacy_availability(['old.csv']).status) == {'unavailable'}


def test_torque_accessor_uses_recorded_final_substep():
    torch=pytest.importorskip('torch')
    from rsl_rl.modules.grf_transition import recorded_applied_torque
    transition=torch.arange(86.).reshape(1,86)
    torque=recorded_applied_torque(transition)
    torch.testing.assert_close(torque,transition[:,:12])
    transition[:,:12] = -1
    assert torque[0,0] == 0
    with pytest.raises(ValueError):
        recorded_applied_torque(torch.zeros(1,24))


def test_frozen_checkpoint_determinism_and_strict_grf_requirement(tmp_path, metadata):
    torch = pytest.importorskip('torch')
    from rsl_rl.modules.actor_critic_pact import ActorCritic_PACT, ContextDecoder
    from reconstruction_eval.models import FrozenModel
    meta = copy.deepcopy(metadata)
    p = meta['policy']
    p.update(actor_layers=[16,16,16],critic_layers=[16,16,16],cenet_enc_layers=[16,16],
             cenet_dec_layers=[16,16,16],grf_dec_layers=[16,16,16])
    actor=ActorCritic_PACT(57,288*5,12,p['actor_layers'],p['critic_layers'],57*20,16,16,p['cenet_enc_layers'],p['activation'])
    decoder=ContextDecoder(32,p['cenet_dec_layers'],276)
    grf=ContextDecoder(44,p['grf_dec_layers'],12)
    path=tmp_path/'test_fixture.pt'
    checkpoint=dict(model_state_dict=actor.state_dict(),decoder_state_dict=decoder.state_dict(),
                    grf_decoder_state_dict=grf.state_dict(),evaluation_metadata=meta,iter=1)
    torch.save(checkpoint,path)
    frozen=FrozenModel(dict(method='pact',checkpoint=str(path)))
    assert not any('checkpoint' in k for k in frozen.identity)
    assert 'iteration' not in frozen.identity
    history=np.zeros((3,20,57),np.float32)
    torque=np.arange(36,dtype=np.float32).reshape(3,12)
    first=frozen.predict(history,torque,meta)
    second=frozen.predict(history,torque,meta)
    for key in first:
        np.testing.assert_equal(first[key],second[key])
    assert all(not p.requires_grad for p in frozen.actor.parameters())
    context=torch.tensor(np.c_[first['latent'],first['explicit']])
    with torch.no_grad():
        expected=frozen.grf_decoder(torch.cat([context,torch.tensor(torque*meta['torque_scale'])],-1)).numpy()/meta['scales']['grf']
    np.testing.assert_allclose(first['grf'],expected)
    checkpoint.pop('grf_decoder_state_dict')
    torch.save(checkpoint,path)
    with pytest.raises(ValueError,match='trained separate GRF'):
        FrozenModel(dict(method='pact',checkpoint=str(path)))


def test_pinocchio_whole_body_reference_and_rotated_base(metadata):
    pin = pytest.importorskip('pinocchio')
    import xml.etree.ElementTree as ET
    from reconstruction_eval.dynamics import Dynamics
    urdf=Path(metadata['urdf'].format(LEGGED_GYM_ROOT_DIR=str(ROOT)))
    tree=ET.parse(urdf)
    links=[]
    for link in tree.findall('link'):
        inertia=link.find('inertial')
        if inertia is None:
            continue
        mass=float(inertia.find('mass').get('value'))
        if mass <= 0:
            continue
        origin=inertia.find('origin')
        xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        quat=pin.Quaternion(pin.rpy.rpyToMatrix(rpy)).coeffs()[[3,0,1,2]]
        I=inertia.find('inertia')
        tensor=np.array([[float(I.get('i'+a+b)) if I.get('i'+a+b) is not None else float(I.get('i'+b+a)) for b in 'xyz'] for a in 'xyz'])
        links.append(dict(name=link.get('name'),mass=mass,com=xyz.tolist(),inertial_quat_wxyz=quat.tolist(),inertia=tensor.tolist()))
    # URDF has a massless `base` fixed to the first massive torso link.
    base=links[0]['name']
    manifest=dict(source_metadata=metadata,physics=dict(links=links,base_link=base,gravity=[0.,0.,-9.81],
        inertia_rule='mass_and_com_shift_fixed_rotational_inertia'))
    dynamics=Dynamics(manifest,urdf)
    nominal_mass=sum(l['mass'] for l in links)
    dynamics.parameters(2.,[.02,-.01,.03])
    mass=pin.computeTotalMass(dynamics.model)
    assert mass == pytest.approx(nominal_mass+2.)
    q=np.zeros(19); q[2]=.4
    q[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(.3,-.2,.5)).coeffs()
    q[7:]=metadata['offsets']['joint_position']
    velocity=np.zeros(18);velocity[0]=1.
    com,p,l=dynamics.momentum(q,velocity)
    np.testing.assert_allclose(p,[mass,0.,0.],atol=1e-10)
    np.testing.assert_allclose(l,np.cross(com,p),atol=1e-10)
    static=np.zeros(18)
    residual=dynamics.residual(q,static,static,.002,np.zeros(12),np.zeros((4,3)))
    np.testing.assert_allclose(residual[:3],[0.,0.,mass*9.81],atol=1e-9)
    support=np.zeros((4,3));support[:,2]=mass*9.81/4
    residual=dynamics.residual(q,static,static,.002,np.zeros(12),support)
    np.testing.assert_allclose(residual[:3],0.,atol=1e-9)
