"""MULE-only reward, per-episode mass, and clipped-observation regressions."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
import torch

from legged_gym.envs.go2.go2_mule.go2_mule_config import resolve_go2_mule
from legged_gym.envs.go2.go2_mule.mule_schema import MULESchema
from legged_gym.envs.go2.go2_mule.go2_mule_isaaclab import IsaacLabMULEEnv
from legged_gym.simulator.isaaclab_simulator_mule import IsaacLabSimulatorMULE
from legged_gym.envs.go2.go2_mule.rolling_payload import sample_offsets, sphere_inertia
from rsl_rl.beyond_robustness_sensors import BRActorObservationCache
from test_mule_rewards_curriculum import fixture, evaluate

TERMS=('collision','dof_pos_limits','feet_air_time','feet_contact_forces')


def test_merge_live_coefficients_shared_formulas_stream_and_dt():
    from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_backend_configs
    from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
    E,T=make_hard_pact_backend_configs('isaaclab'); shared=E()
    for i,k in enumerate(TERMS): setattr(shared.rewards.scales,k,(-.3 if k!='feet_air_time' else .3)*(i+1))
    spec=resolve_go2_mule(shared,T()).shared.mule_reward
    e,s=fixture(additions=True); e.spec=spec; e.spec['curriculum']['enabled']=False
    s.dof_pos[:,0]=1.25; s.link_contact_forces[:,2,0]=2
    s.link_contact_forces[:,s.feet_contact_indices,2]=e.cfg.rewards.max_contact_force+7
    evaluate(e,s)
    context=NS(simulator=s,cfg=e.cfg)
    for k in TERMS:
        coefficient=getattr(shared.rewards.scales,k)
        assert spec['terms'][k]['coefficients']==dict(nominal=coefficient,adaptive=0.)
        assert k not in e.last_weighted['adaptive']
        if k!='feet_air_time':
            torch.testing.assert_close(e.last_raw['nominal'][k],getattr(Go2PACT,'_reward_'+k)(context))
        torch.testing.assert_close(e.last_weighted['nominal'][k],e.last_raw['nominal'][k]*coefficient*.02)
    assert (e.last_raw['nominal']['dof_pos_limits']==.25).all() # excess, NOT BR count


def test_air_time_landing_filter_partial_reset_and_fresh_history():
    e,s=fixture(additions=True); e.spec['curriculum']['enabled']=False
    e.spec['terms']['feet_air_time']['coefficients']['nominal']=.7
    # Call real evaluator with nonzero command (fixture defaults to standstill).
    def step():
        n=3; z=torch.zeros(n,12); commands=torch.zeros(n,4); commands[:,0]=1
        return e.evaluate(s,NS(nominal_raw=z,adaptive_raw=z),z,commands,
            NS(force_b=torch.zeros(n,4,3),valid=torch.ones(n,4,dtype=torch.bool)),torch.ones(n)*12,
            torch.zeros(n,4),torch.zeros(n,dtype=torch.bool),torch.zeros(n,dtype=torch.bool),torch.ones(n,dtype=torch.bool))
    for _ in range(30): step()
    untouched=e.air[1:].clone(); e.reset(torch.tensor([0]),s)
    torch.testing.assert_close(e.air[1:],untouched)
    s.link_contact_forces[:,s.feet_contact_indices,2]=2
    step()
    torch.testing.assert_close(e.last_raw['nominal']['feet_air_time'],torch.tensor([0.,4*(.62-.5),4*(.62-.5)]))
    assert e.air.sum()==0 and e.contacts.all()
    s.link_contact_forces.zero_(); step() # previous-contact filter suppresses one-frame drop
    assert e.air.sum()==0
    fresh,_=fixture(additions=True); fresh.spec=deepcopy(e.spec); fresh.load_state_dict(e.state_dict())
    assert fresh.air.sum()==0 and not fresh.contacts.any()


def test_reward_snapshot_uses_ground_not_payload_support():
    e,s=fixture(additions=True)
    s.hard_pact_executed_torque=lambda:s.torques
    s.last_requested_torque=s.requested_torque
    s.ground_total_forces_world=torch.zeros_like(s.link_contact_forces)
    s.link_contact_forces[:,2,2]=1000 # unfiltered tray support must NOT be collision
    env=object.__new__(IsaacLabMULEEnv); env.simulator=s; env.rewards=None
    env.reward_collision_ids=s.penalised_contact_indices
    snapshot=env._reward_snapshot(); snapshot.near_edge=s.near_edge
    evaluate(e,snapshot); assert e.last_raw['nominal']['collision'].sum()==0
    s.ground_total_forces_world[:,2,2]=5
    snapshot=env._reward_snapshot(); snapshot.near_edge=s.near_edge
    evaluate(e,snapshot); assert (e.last_raw['nominal']['collision']==1).all()


def test_cadence_ineligible_reset_resamples_mass_but_resume_preserves_it():
    c=resolve_go2_mule().payload; n=3
    sim=object.__new__(IsaacLabSimulatorMULE); sim.payload_cfg=c; sim.payload_progress=1.; sim._device='cpu'
    sim.payload_rng=torch.Generator().manual_seed(42)
    sim.payload_reaction_world=torch.zeros(n,3); sim.ball_radius=torch.full((n,4),.04)
    sim.ball_mass=torch.full((n,4),1.9); sim.ball_friction=torch.full((n,4),.4); sim.ball_restitution=torch.zeros(n,4)
    sim.payload_active=torch.ones(n,dtype=torch.bool); sim.realized_severity=torch.ones(n)
    sim.ball_offsets=sample_offsets(c,sim.ball_radius,sim.payload_rng,1.)
    sim.ball_relative_velocity=torch.zeros(n,4,3); sim.ball_relative_omega=torch.zeros(n,4,3)
    sim.mass_elapsed=torch.full((n,),3.5); sim.mass_change_count=torch.full((n,),7)
    sim._reset_sample_ids=lambda group,ids:ids[:0] # persistent cadence NOT due
    sim._uniform=lambda shape,lo,hi:lo+(hi-lo)*torch.rand(shape,generator=sim.payload_rng)
    sim._robot=NS(data=NS(root_link_quat_w=torch.tensor([[1.,0,0,0]]*n),root_link_pos_w=torch.zeros(n,3),
        root_link_lin_vel_w=torch.zeros(n,3),root_link_ang_vel_w=torch.zeros(n,3)))
    class View:
        def __init__(self): self.mass=torch.full((n,1),1.9); self.inertia=torch.zeros(n,9); self.material=torch.zeros(n,1,3); self.disabled=torch.zeros(n,1)
        def get_masses(self): return self.mass
        def set_masses(self,x,ids): self.mass[ids]=x[ids]
        def get_inertias(self): return self.inertia
        def set_inertias(self,x,ids): self.inertia[ids]=x[ids]
        def get_material_properties(self): return self.material
        def set_material_properties(self,x,ids): self.material[ids]=x[ids]
        def get_disable_gravities(self): return self.disabled
        def set_disable_gravities(self,x,ids): self.disabled[ids]=x[ids]
    sim.balls=[NS(root_physx_view=View(),write_root_state_to_sim=lambda x,ids:None,reset=lambda ids:None) for _ in range(4)]
    sim.ball_sensors=sim._ball_contacts=[NS(reset=lambda ids:None)]*4
    ids=torch.tensor([0,2]); sim._resume_samples=True
    sim.reset_payload(ids)
    assert (sim.ball_mass==1.9).all() and (sim.mass_elapsed==3.5).all()
    sim._resume_samples=False # adapter clears this after the initial fresh reset
    sim.reset_payload(ids)
    assert (sim.ball_mass[ids]<=c.reset_mass_range[1]).all() and (sim.ball_mass[ids]>=c.mass_floor_kg).all()
    assert (sim.ball_mass[1]==1.9).all() and sim.mass_elapsed[1]==3.5 and sim.mass_change_count[1]==7
    assert sim.mass_elapsed[ids].sum()==sim.mass_change_count[ids].sum()==0
    assert (sim.ball_friction[ids]==.4).all()
    for k,ball in enumerate(sim.balls):
        torch.testing.assert_close(ball.root_physx_view.get_inertias()[ids],sphere_inertia(sim.ball_mass[ids,k],sim.ball_radius[ids,k]).reshape(-1,9))


def test_shared_clip_cached_after_noise_and_physical_labels_unchanged():
    c=resolve_go2_mule(environment_overrides={'env.num_envs':3,'normalization.clip_observations':.25})
    assert c.environment.schema==c.training.schema and c.training.schema.clip_observations==.25
    from test_mule_learning import SyntheticMULE
    synthetic=SyntheticMULE(c.training); original=synthetic.observe()
    cache=BRActorObservationCache(3,'cpu',c.shared.noise,c.training.schema.clip_observations,17)
    clean=torch.full((3,45),10.); clean[:,::2]=-10
    cache.refresh(clean)
    assert torch.isfinite(cache.values).all() and cache.values.abs().max()==.25
    before=cache.values.clone()
    cache.refresh(clean,torch.tensor([0]))
    torch.testing.assert_close(cache.values[1:],before[1:])
    # Exercise the actual adapter refresh: only deployable observations clip,
    # not physical velocity supervision or the clean privileged state.
    env=object.__new__(IsaacLabMULEEnv); env.cfg=c.environment; env.input_schema=c.environment.schema
    env.commands=torch.zeros(3,4); env.previous_action=torch.zeros(3,12)
    env.actor_observation_cache=cache; env.episode_id=torch.zeros(3,dtype=torch.long)
    env.clock=0; env._packet=None
    env.simulator=NS(_robot=NS(data=NS(root_link_quat_w=torch.tensor([[1.,0,0,0]]*3),
        root_link_pos_w=torch.zeros(3,3),root_link_ang_vel_w=torch.ones(3,3)*10,
        root_link_lin_vel_w=torch.ones(3,3)*7)),_global_gravity=torch.tensor([[0.,0,-1.]]*3),
        dof_pos=torch.ones(3,12)*3,dof_vel=torch.ones(3,12)*9,payload_reaction_world=torch.zeros(3,3),
        _measured_heights=torch.zeros(3,c.training.schema.terrain_dim),
        hard_pact_executed_torque=lambda:torch.ones(3,12),
        leg_jacobians_b=lambda:torch.eye(3).expand(3,4,3,3))
    env._refresh()
    assert env.observe().actor.abs().max()<=.25
    torch.testing.assert_close(env.observe().velocity_b,torch.full((3,3),7.))
    assert env.observe().critic.abs().max()>.25
    rng=cache.generator.get_state().clone(); env.observe(); env.observe()
    torch.testing.assert_close(rng,cache.generator.get_state())
    from rsl_rl.runners.mule_runner import MULERunner
    config=replace(c.training,num_steps_per_env=2,num_mini_batches=1,num_learning_epochs=1,auxiliary_epochs=1)
    raw_observe=synthetic.observe
    seen_time=torch.full((3,),-1); seen_episode=torch.full((3,),-1)
    def observe():
        packet=raw_observe()
        ids=((seen_time!=packet.timestamp)|(seen_episode!=packet.episode_id)).nonzero().flatten()
        cache.refresh(packet.actor*100,ids)
        seen_time[ids]=packet.timestamp; seen_episode[ids]=packet.episode_id[ids]
        return replace(packet,actor=cache.values.clone())
    synthetic.observe=observe
    r=MULERunner(synthetic,config); r.collect_rollout()
    # Real storage's reconstruction targets and histories both originate in packets.
    assert r.history.observations.abs().max()<=.25
    assert all(r.alg.storage.data[k].abs().max()<=.25 for k in ('history','next_observation'))
    torch.testing.assert_close(r.alg.storage.data['next_observation'][-1],synthetic.last_terminal.actor)
    torch.testing.assert_close(original.velocity_b,torch.zeros(3,3))


def test_old_or_changed_clip_contract_rejected_before_mutation(tmp_path):
    from test_mule_learning import runner, assert_equal
    r=runner(); r.learn(1); path=tmp_path/'source.pt'; r.save(path)
    target=runner(); before=deepcopy(target.alg.model.state_dict())
    target.config=replace(target.config,schema=replace(target.config.schema,clip_observations=3.))
    with pytest.raises(ValueError): target.load(path)
    assert_equal(before,target.alg.model.state_dict())
    with pytest.raises(ValueError,match='schema'): MULESchema(version=1)
    with pytest.raises(ValueError,match='clip_observations'): MULESchema(clip_observations=float('inf'))
    old=deepcopy(r.config.contract()); old['schema']['version']=1
    old['schema'].pop('clip_observations')
    assert not r.config.accepts_model_contract(old)


def test_production_resume_rejects_unpickled_old_observation_schema(tmp_path):
    from test_mule_production import production
    from legged_gym.scripts.train_mule import parser,resolve_run
    _,r=production(); path=tmp_path/'source.pt'; r.save(path)
    saved=torch.load(path,weights_only=False)
    object.__setattr__(saved['resolved_configuration'].training.schema,'version',1)
    torch.save(saved,path)
    with pytest.raises(ValueError,match='schema'):
        resolve_run(parser().parse_args(['--checkpoint',str(path)]))
