"""Stage-4 configuration/sensor/reset contracts; no simulator initialization."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
import torch
from rsl_rl.beyond_robustness_config import resolve_beyond_robustness
from rsl_rl.beyond_robustness_schema import BeyondRobustnessInputSchema
from rsl_rl.beyond_robustness_sensors import BRActorObservationCache
from rsl_rl.beyond_robustness_payload import payload_geometry,cuboid_inertia,SlidingPayloadConfig,spawn_state,load_labels
from rsl_rl.modules.actor_critic_beyond_robustness import ActorCriticBeyondRobustness
from rsl_rl.algorithms.ppo_beyond_robustness import PPOBeyondRobustness
from legged_gym.envs.go2.go2_beyond_robustness import IsaacLabBeyondRobustnessEnv as Env
from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_backend_configs


def test_live_factory_and_isolation():
    a=resolve_beyond_robustness(); b=resolve_beyond_robustness()
    s=a.training.input_schema
    assert (s.observation_dim,s.state_dim,s.parameter_dim)==(45,227,89)
    E,T=make_hard_pact_backend_configs('isaaclab'); e,t=E(),T()
    assert a.training.num_envs==e.env.num_envs==4096
    assert a.sim_params['dt']==e.control.dt/e.control.decimation
    assert a.training.num_steps_per_env==t.runner.num_steps_per_env
    assert a.training.history_length==e.env.num_obs_hist
    assert a.training.max_iterations==t.runner.max_iterations and a.training.save_interval==t.runner.save_interval
    a.environment.control.stiffness['joint']=17
    a.environment.terrain.measured_points_x.append(2.)
    assert b.environment.control.stiffness==e.control.stiffness
    assert b.environment.terrain.measured_points_x==e.terrain.measured_points_x
    assert not hasattr(a.training,'hard_pact_qp') and not hasattr(a.environment,'physics')


def test_changed_shared_settings_reach_model_optimizer_schema_and_noise():
    E,T=make_hard_pact_backend_configs('isaaclab'); e,t=E(),T()
    e.env.num_envs=3; e.env.num_obs_hist=7; e.env.episode_length_s=9.
    e.control.dt=.03; e.control.decimation=3; e.control.action_scale=.4
    e.control.stiffness={'joint':41.}; e.control.damping={'joint':1.2}
    e.terrain.measured_points_x=[-.2,.2]; e.terrain.measured_points_y=[0.]
    e.noise.noise_level=.7; e.noise.noise_scales.ang_vel=.8
    t.seed=73; t.runner.max_iterations=31; t.runner.save_interval=7
    t.algorithm.learning_rate=8e-4; t.algorithm.auxiliary_learning_rate=6e-4
    t.policy.init_noise_std=[.6]*12+[.8]*12
    c=resolve_beyond_robustness(e,t); train=c.training
    assert train.input_schema.state_dim==86 and train.history_length==7
    assert c.sim_params['dt']==.01 and c.environment.control.action_scale==.4
    assert c.environment.control.stiffness['joint']==41. and c.environment.env.episode_length_s==9.
    from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT
    controller=NS(_cfg=c.environment,_num_actions=12,_kp_scale=torch.ones(3,12),_kd_scale=torch.ones(3,12),
        _p_gains=torch.full((12,),41.),_d_gains=torch.full((12,),1.2),
        _motor_strength=torch.full((3,12),.9),default_dof_pos=torch.full((12,),.1),
        dof_pos=torch.full((3,12),.2),dof_vel=torch.full((3,12),.3),first_loop=True)
    torque=IsaacLabSimulator_PACT._compute_torques(controller,torch.ones(3,12))
    torch.testing.assert_close(torque,torch.full((3,12),.9*(41*(.4+.1-.2)-1.2*.3)))
    model=ActorCriticBeyondRobustness(train); alg=PPOBeyondRobustness(model,train)
    assert model.history_dim==7*45 and model.latent_dim==32 and model.num_actions==12
    assert model.load_estimator[-1].out_features==7
    assert [m.out_features for m in model.actor if isinstance(m,torch.nn.Linear)]==[512,256,128,12]
    assert model.std.eq(.6).all()
    assert alg.optimizer.param_groups[0]['lr']==8e-4
    assert alg.load_optimizer.param_groups[0]['lr']==6e-4
    assert train.use_adaptive_entropy==t.algorithm.use_adaptive_entropy
    cache=BRActorObservationCache(3,'cpu',c.environment.noise,100,train.seed)
    torch.testing.assert_close(cache.amplitude[:3],torch.full((3,),.56))
    assert c.payload.seed==train.seed==73


def test_cached_noise_partial_reset_and_rng_restore():
    c=resolve_beyond_robustness(); cache=BRActorObservationCache(4,'cpu',c.environment.noise,100,13)
    clean=torch.arange(45.).repeat(4,1)/10
    cache.refresh(clean); original=cache.values.clone(); rng=cache.generator.get_state().clone()
    for _ in range(3):
        torch.testing.assert_close(cache.values,original,rtol=0,atol=0)
    assert torch.equal(rng,cache.generator.get_state())
    for sl in (slice(6,9),slice(33,45)):
        torch.testing.assert_close(cache.values[:,sl],clean[:,sl],rtol=0,atol=0)
    cache.refresh(clean,torch.tensor([1]))
    torch.testing.assert_close(cache.values[[0,2,3]],original[[0,2,3]],rtol=0,atol=0)
    assert not torch.equal(cache.values[1],original[1])
    state=cache.state_dict(); cache.refresh(clean); expected=cache.values.clone()
    cache.load_state_dict(state); cache.refresh(clean)
    torch.testing.assert_close(cache.values,expected,rtol=0,atol=0)
    noise=cache.noise.clone(); clean[:,6]=5.; cache.refresh(clean,redraw=False)
    torch.testing.assert_close(cache.noise,noise,rtol=0,atol=0)
    cfg=deepcopy(c.environment.noise); cfg.add_noise=False
    disabled=BRActorObservationCache(4,'cpu',cfg,1.,13); state=disabled.state_dict()
    disabled.refresh(clean)
    assert torch.equal(disabled.generator.get_state(),state['rng'])
    torch.testing.assert_close(disabled.values,clean.clamp(-1,1))


def test_shared_randomized_robot_resets_and_selected_isolation():
    c=resolve_beyond_robustness(); env=Env.__new__(Env); env.cfg=c.environment
    env.device='cpu'; env.num_actions=12
    n=4; q=torch.zeros(n,12); pos=torch.zeros(n,3); lin=torch.zeros_like(pos); ang=lin.clone()
    def joints(ids,x,v): q[ids]=x; assert v.eq(0).all()
    def root(ids,x,quat,v,w): pos[ids]=x; lin[ids]=v; ang[ids]=w
    env.simulator=NS(default_dof_pos=torch.ones(12),custom_origins=True,
        base_init_pos=torch.tensor([0.,0.,.44]),base_init_quat=torch.tensor([0.,0.,0.,1.]),
        env_origins=torch.zeros(n,3),reset_dofs=joints,reset_root_states=root)
    torch.manual_seed(17); ids=torch.tensor([1,3]); env._reset_dofs(ids); env._reset_root_states(ids)
    assert (q[ids]-1).abs().max()<=.3 and (q[ids]-1).abs().sum()>0
    assert pos[ids,:2].abs().max()<=.5 and lin[ids].abs().max()<=.5 and ang[ids].abs().max()<=.5
    assert q[[0,2]].eq(0).all() and pos[[0,2]].eq(0).all()


def test_terrain_delay_absolute_iteration_and_per_reset_not_per_iteration():
    env=Env.__new__(Env); env.cfg=resolve_beyond_robustness().environment
    env.cfg.terrain.curriculum_upward_delay_iterations=12
    mask=torch.tensor([True,False])
    env.begin_iteration(11); assert not env._filter_terrain_move_up(mask).any()
    env.begin_iteration(12); assert torch.equal(env._filter_terrain_move_up(mask),mask)
    env.begin_iteration(100); assert torch.equal(env._filter_terrain_move_up(mask),mask)
    # The callback is the shared reset-triggered implementation, not a new cadence.
    from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
    assert Env._update_terrain_curriculum is Go2PACT._update_terrain_curriculum
    env.last_completed_iteration=0; env.command_streak=3
    env.cfg.commands.curriculum=True; env.cfg.commands.curriculum_patience_iterations=4
    env.cfg.commands.curriculum_threshold=.5
    env.simulator=NS(use_domainrand_curriculum=False)
    env.domain_rand_curriculum=NS(progress={})
    expanded=[]; env._expand_command_curriculum_ranges=lambda:expanded.append(True)
    env.end_iteration(2,.9,10); assert env.command_streak==1 and not expanded # skipped iteration resets streak
    env.end_iteration(3,float('nan'),10); assert env.command_streak==0


def test_seeded_geometry_inertia_and_clearance():
    cfg=SlidingPayloadConfig(seed=91); bank,ids=payload_geometry(cfg,37)
    other,other_ids=payload_geometry(cfg,37)
    torch.testing.assert_close(bank,other,rtol=0,atol=0); assert torch.equal(ids,other_ids)
    size=bank[ids]; assert size.unique(dim=0).shape[0]==16
    assert not torch.equal(bank,payload_geometry(replace(cfg,seed=92),37)[0])
    assert (size>=torch.tensor(cfg.box_size_min)).all() and (size<=torch.tensor(cfg.box_size_max)).all()
    mass=torch.linspace(.2,3.,37); inertia=cuboid_inertia(mass,size)
    torch.testing.assert_close(inertia[:,0,0],mass*(size[:,1]**2+size[:,2]**2)/12)
    usable=torch.tensor(cfg.tray_size[:2])/2-cfg.wall_thickness-size[:,:2]/2-cfg.spawn_clearance
    assert (usable>0).all() and (torch.linalg.eigvalsh(inertia)>0).all()


def test_validation_and_legacy_schema_rejection():
    for overrides in ({'sim.dt':.123},{'control.dt':0.},{'control.decimation':0},
                      {'control.stiffness':{'missing':1.}}, {'env.num_actions':24}):
        with pytest.raises(ValueError): resolve_beyond_robustness(environment_overrides=overrides)
    with pytest.raises(ValueError,match='Incompatible BR observation schema'):
        BeyondRobustnessInputSchema(version=2)
    with pytest.raises(ValueError,match='seed'):
        resolve_beyond_robustness(payload_overrides={'seed':-10})


def test_large_world_coordinates_do_not_manufacture_reset_load_velocity():
    q=torch.tensor([[1.,0,0,0.]])
    origin=torch.tensor([[123.,-87.,3.]])
    velocity=torch.tensor([[.2,-.4,.1]]); omega=torch.tensor([[.5,.4,.3]])
    state=spawn_state(q,origin,velocity,omega,torch.tensor([[.011,.019,.153]]),torch.zeros(1,3))
    labels=load_labels(q,origin,velocity,omega,state[:,:3],state[:,7:10],torch.ones(1))
    torch.testing.assert_close(labels[:,3:6],torch.zeros(1,3),rtol=0,atol=5e-8)


def test_shared_command_deadband_heading_and_zero_patience_cadence():
    env=Env.__new__(Env); env.cfg=resolve_beyond_robustness().environment
    env.device='cpu'; env.commands=torch.zeros(2,4); env.forward_vec=torch.tensor([[1.,0,0]]).repeat(2,1)
    env.simulator=NS(base_quat=torch.tensor([[0.,0,0,1.]]).repeat(2,1))
    env.command_ranges={'lin_vel_x':[.1,.1],'lin_vel_y':[.1,.1],
        'ang_vel_yaw':[.05,.05],'heading':[1.,1.]}
    env.cfg.commands.heading_command=False
    env._sample_commands(torch.arange(2))
    assert env.commands[:,:3].eq(0).all() # joint xy/yaw norm, NOT an xy-only gate
    env.command_ranges['ang_vel_yaw']=[.3,.3]
    env._sample_commands(torch.arange(2))
    torch.testing.assert_close(env.commands[0,:3],torch.tensor([.1,.1,.3]))
    env.cfg.commands.heading_command=True; env.command_ranges['ang_vel_yaw']=[-.25,.25]
    env._sample_commands(torch.arange(2)); assert env.commands[:,2].eq(.25).all()
    env.init_done=True; env.max_episode_length=10; env.episode_tracking=torch.tensor([[9.,0],[9.,0]])
    env.cfg.commands.curriculum=True; env.cfg.commands.curriculum_patience_iterations=0
    env.clock=9; before=deepcopy(env.command_ranges)
    env._reset_command_curriculum(torch.arange(2)); assert env.command_ranges==before
    env.clock=10; env._reset_command_curriculum(torch.arange(2))
    assert env.command_ranges['lin_vel_x'][1]==.6
    before=deepcopy(env.command_ranges); env.cfg.commands.curriculum_patience_iterations=1
    env._reset_command_curriculum(torch.arange(2)); assert env.command_ranges==before
