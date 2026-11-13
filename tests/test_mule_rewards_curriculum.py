"""Production MULE rewards and shared gates, without simulator construction."""
from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
import torch
from legged_gym.envs.go2.go2_mule.go2_mule_config import resolve_go2_mule
from legged_gym.envs.go2.go2_mule.mule_rewards import MULERewards,NATIVE,ADDITIONS,grf_reward
from legged_gym.envs.go2.go2_mule.go2_mule_isaaclab import IsaacLabMULEEnv,refresh_after_push
from legged_gym.envs.go2.go2_mule.rolling_payload import RollingPayloadConfig
from legged_gym.envs.go2.go2_hard_pact.domain_rand_curriculum import HardPACTDomainRandCurriculum


def fixture(n=3,additions=False,dt=.02):
    from test_beyond_robustness_rewards import fixture as shared_fixture
    _,s=shared_fixture(n)
    cfg=resolve_go2_mule(environment_overrides={'env.num_envs':n,'control.dt':dt},
        reward_overrides={'additional_coefficients':{} if additions else {k:0. for k in ADDITIONS}})
    e=MULERewards(cfg.shared.mule_reward,cfg.shared,n,'cpu'); e.reset(torch.arange(n),s)
    return e,s


def evaluate(e,s,nominal=None,adaptive=None,valid=None):
    n=len(s.dof_pos); a=torch.zeros(n,12)
    cmd=NS(nominal_raw=a if nominal is None else nominal,adaptive_raw=a if adaptive is None else adaptive)
    f=NS(force_b=torch.zeros(n,4,3),valid=torch.ones(n,4,dtype=torch.bool))
    return e.evaluate(s,cmd,a,torch.zeros(n,4),f,torch.ones(n)*12,torch.zeros(n,4),
        torch.zeros(n,dtype=torch.bool),torch.zeros(n,dtype=torch.bool),
        torch.ones(n,dtype=torch.bool) if valid is None else valid)


def test_native_coefficients_zeros_dt_and_no_phase_dependence():
    assert list(NATIVE.values())==[(1.,0.),(.5,0.),(-2.,-2.),(-.05,-.05),(-.2,-.2),
        (-2.5e-7,-2.5e-7),(-2e-5,0.),(-2.,-2.),(-.01,-.01),(-.001,-.01),(-.01,-.01),(0.,2.)]
    e,s=fixture(); n,a=evaluate(e,s)
    torch.testing.assert_close(n,torch.full((3,),.02*1.49))
    torch.testing.assert_close(a,torch.full((3,),-.02*.01))
    e2,s2=fixture(dt=.04); n2,a2=evaluate(e2,s2)
    torch.testing.assert_close(n2,2*n); torch.testing.assert_close(a2,2*a)
    e.begin_iteration(4000); n3,_=evaluate(e,s,adaptive=torch.ones(3,12)*20)
    torch.testing.assert_close(n,n3) # native nominal unchanged by adaptive action/phase
    assert 'tracking_lin_vel' not in e.last_weighted['adaptive'] and 'joint_power' not in e.last_weighted['adaptive']
    assert 'grf_tracking' not in e.last_weighted['nominal']


def test_shared_formulas_and_explicit_addition_contract():
    from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
    e,s=fixture(additions=True); s.dof_vel[:]=2; s.feet_vel[:,:,0]=1; s.torques[:]=3
    s.base_lin_vel[:]=torch.tensor([.4,.2,.1]); s.base_ang_vel[:]=torch.tensor([.2,.1,.3])
    s.projected_gravity[:,0]=.2
    previous=e.velocity.clone(); evaluate(e,s)
    context=NS(simulator=s,cfg=e.cfg,commands=torch.zeros(3,4),dt=e.dt); s.last_dof_vel=previous
    for k in NATIVE.keys()-{'grf_tracking','action_rate','action_smoothness'}:
        torch.testing.assert_close(e.last_raw['nominal'][k],getattr(Go2PACT,'_reward_'+k)(context))
    for k in ADDITIONS:
        assert e.spec['terms'][k]['coefficients']['adaptive']==0
    assert not {'orientation','foot_clearance_terrain_aware','joint_power_dist','pbrs_height'}&ADDITIONS.keys()
    assert e.spec['excluded_shared']['torque_cancellation']
    with pytest.raises(ValueError,match='protected'):
        resolve_go2_mule(reward_overrides={'additional_coefficients':{'tracking_lin_vel':8}})
    with pytest.raises(ValueError,match='protected'):
        resolve_go2_mule(reward_overrides={'curriculum':{'bounds':{'orientation':[-1,0]}}})


def test_grf_strict_thresholds_mass_and_invalid_estimates():
    h=torch.tensor([.31,.29,.29,.30,.29,.29]); f=torch.zeros(6,4,3); f[:,:,2]=30
    valid=torch.ones(6,4,dtype=torch.bool); valid[4,0]=False; f[5,0,0]=float('nan')
    robot=torch.ones(6)*10.25 # includes 250g tray once
    balls=torch.ones(6,4)*.25; balls[2]=2.
    r=grf_reward(h,.3,f,valid,robot,balls)
    torch.testing.assert_close(r,torch.tensor([.75,.5,0,0,0,0]))
    f[1,:,2]=(11.25*9.81)/4
    assert grf_reward(h,.3,f,valid,robot,balls)[1]==0 # equality not >


def test_stream_histories_reset_and_invalid_rows():
    e,s=fixture(); a=torch.ones(3,12)
    evaluate(e,s,a,2*a); evaluate(e,s,2*a,4*a)
    torch.testing.assert_close(e.last_raw['nominal']['action_rate'],torch.full((3,),12.))
    torch.testing.assert_close(e.last_raw['adaptive']['action_rate'],torch.full((3,),48.))
    evaluate(e,s,4*a,9*a)
    torch.testing.assert_close(e.last_raw['nominal']['action_smoothness'],torch.full((3,),12.))
    torch.testing.assert_close(e.last_raw['adaptive']['action_smoothness'],torch.full((3,),108.))
    before=e.actions[1:].clone(); e.reset(torch.tensor([0]),s)
    torch.testing.assert_close(before,e.actions[1:]); assert e.age[0]==0
    s.dof_vel[1]=float('nan')
    n,b=evaluate(e,s,valid=torch.tensor([True,False,True]))
    assert n[1]==b[1]==0 and torch.isfinite(n).all() and torch.isfinite(b).all()
    assert e.last_raw['nominal']['action_rate'][0]==0


def curriculum_env():
    cfg=resolve_go2_mule(environment_overrides={
        'domain_rand.use_joint_dynamics_curriculum':False,'domain_rand.use_mass_com_curriculum':False,
        'domain_rand.use_disturbance_curriculum':True,'domain_rand.disabled_curriculum_use_final_bounds':False,
        'domain_rand.push_warmup':0,'domain_rand.step_interval':1,'domain_rand.disturbance_progress_delta':.25,
        'domain_rand.min_reward_to_step':.5,'domain_rand.reward_ema_alpha':1.,
        'commands.curriculum_patience_iterations':2})
    e=object.__new__(IsaacLabMULEEnv); e.shared=cfg.shared; e.domain_rand_curriculum=HardPACTDomainRandCurriculum(cfg.shared)
    e.simulator=NS(payload_cfg=cfg.payload,payload_progress=0.,balls=[1],use_domainrand_curriculum=True)
    e.simulator.set_payload_severity=lambda value:setattr(e.simulator,'payload_progress',value)
    e.command_ranges={k:list(getattr(cfg.shared.commands.ranges,k)) for k in ('lin_vel_x','lin_vel_y','ang_vel_yaw')}
    e.last_completed_iteration=-1; e.command_streak=0; e.curriculum_metrics={}
    return e


def test_curriculum_missing_gate_once_boundary_and_resume():
    e=curriculum_env(); assert e.requested_optimization_phase() is None
    evidence=dict(lin_vel_tracking=.9,valid_count=120)
    e.end_iteration(0,evidence); assert e.simulator.payload_progress==0 # shared warmup
    e.end_iteration(1,None); assert e.simulator.payload_progress==0
    e.end_iteration(2,dict(lin_vel_tracking=float('nan'),valid_count=120)); assert e.command_streak==0
    e.end_iteration(3,evidence); assert e.simulator.payload_progress==.25
    assert e.requested_optimization_phase()=='joint_adaptation'
    e.end_iteration(3,evidence); assert e.simulator.payload_progress==.25
    other=curriculum_env(); other.domain_rand_curriculum.load_state_dict(e.domain_rand_curriculum.state_dict())
    other.last_completed_iteration=e.last_completed_iteration; other.command_streak=e.command_streak
    other.command_ranges=deepcopy(e.command_ranges)
    for i,value in enumerate((.2,.9,.9),4):
        x=dict(lin_vel_tracking=value,valid_count=20)
        e.end_iteration(i,x); other.end_iteration(i,x)
        assert e.domain_rand_curriculum.progress==other.domain_rand_curriculum.progress
        assert e.command_ranges==other.command_ranges and e.command_streak==other.command_streak
    e.shared.mule.auto_joint_adaptation=False; assert e.requested_optimization_phase() is None


def test_reward_schedule_resume_and_native_protection():
    e,s=fixture(additions=True); e.begin_iteration(731)
    state=e.state_dict(); other,_=fixture(additions=True); other.load_state_dict(state)
    assert e.coefficients()==other.coefficients()
    for k,weights in NATIVE.items(): assert tuple(e.coefficients()[k].values())==weights
    assert other.age.sum()==0


def test_push_refresh_and_repeated_observation_timestamp():
    s=NS(_robot=NS(data=NS(root_link_quat_w=torch.tensor([[1.,0,0,0]]),root_link_pos_w=torch.zeros(1,3),
        root_link_lin_vel_w=torch.tensor([[1.,2.,3.]]),root_link_ang_vel_w=torch.tensor([[0.,0.,2.]]))),
        _base_world_lin_vel=torch.zeros(1,3),_base_world_ang_vel=torch.zeros(1,3),
        _base_lin_vel=torch.zeros(1,3),_base_ang_vel=torch.zeros(1,3),_feet_vel=torch.zeros(1,4,3),
        feet_pos=torch.tensor([[[1.,0,0]]*4]))
    refresh_after_push(s)
    torch.testing.assert_close(s._feet_vel,torch.tensor([[[1.,4.,3.]]*4]))
    torch.testing.assert_close(s._base_lin_vel,s._robot.data.root_link_lin_vel_w)
    before=s._feet_vel.clone(); refresh_after_push(s); torch.testing.assert_close(before,s._feet_vel)


def test_count_weighted_runner_evidence_and_boundary_hook():
    from test_mule_learning import runner
    from dataclasses import replace
    r=runner(); old=r.env.step; index=[0]
    def step(command):
        result=old(command); i=index[0]; index[0]+=1
        return replace(result,performance_metrics=dict(lin_vel_tracking=float(i),ang_vel_tracking=.5,terrain_level=1.,valid_count=i+1))
    r.env.step=step; r.env.requested_optimization_phase=lambda:'joint_adaptation'
    _,p=r.collect_rollout(); n=r.config.num_steps_per_env
    assert p['valid_count']==n*(n+1)/2
    assert p['lin_vel_tracking']==pytest.approx(sum(i*(i+1) for i in range(n))/p['valid_count'])
    assert r.alg.model.phase=='joint_adaptation'


def test_friction_endpoints_and_generation_forces_next_reset():
    from legged_gym.simulator.reset_randomization_cadence import ResetRandomizationCadence
    c=RollingPayloadConfig(); assert c.friction_at(0)==c.light_friction_range
    assert c.friction_at(1)==pytest.approx(c.friction_range)
    cadence=ResetRandomizationCadence(2,'cpu',100); ids=torch.arange(2)
    cadence.update_ranges({'mule_payload':(0.,)}); cadence.begin_reset(ids)
    assert len(cadence.select('mule_payload',ids))==2
    cadence.begin_reset(ids); assert len(cadence.select('mule_payload',ids))==0
    cadence.update_ranges({'mule_payload':(.25,)})
    assert cadence.select('mule_payload',ids[:1]).tolist()==[0]
    assert cadence.seen['mule_payload'][1]!=cadence.generations['mule_payload']
