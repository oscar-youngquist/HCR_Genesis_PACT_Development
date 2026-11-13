"""Simulator-free reward contract, timing and isolated shared additions."""
from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
import torch
from rsl_rl.beyond_robustness_config import resolve_beyond_robustness
from rsl_rl.beyond_robustness_rewards import BRRewards, ORIGINAL, ADDITIONS, tracking


def fixture(n=3, additions=False, dt=.02):
    resolved=resolve_beyond_robustness(environment_overrides={'env.num_envs':n,'control.dt':dt},
        reward_overrides={'additional_coefficients':{} if additions else {k:0. for k in ADDITIONS}})
    z=lambda *shape: torch.zeros(n,*shape)
    feet=z(4,3); feet[:,:,:2]=torch.tensor([[.2,-.15],[.2,.15],[-.2,-.15],[-.2,.15]])
    s=NS(base_pos=z(3),base_quat=z(4),projected_gravity=z(3),dof_pos=z(12),dof_vel=z(12),
        default_dof_pos=z(12),dof_pos_limits=torch.tensor([[-1.,1.]]*12),
        dof_pos_limits_hard=torch.tensor([[-1.,1.]]*12),torque_limits=torch.ones(12)*20,
        feet_pos=feet,feet_vel=z(4,3),measured_heights=z(143),_height_around_feet=z(4,9),
        _max_height_ahead_feet=z(4),_normal_vector_around_feet=z(12),near_edge=z(4).bool(),
        base_lin_vel=z(3),base_ang_vel=z(3),torques=z(12),requested_torque=z(12),
        link_contact_forces=z(7,3),feet_contact_indices=[3,1,5,0],feet_indices=[0,1,2,3],penalised_contact_indices=[2,4,6])
    s.base_pos[:,2]=resolved.environment.rewards.base_height_target; s.base_quat[:,3]=1; s.projected_gravity[:,2]=-1
    engine=BRRewards(resolved.environment.br_reward,resolved.environment,n,'cpu')
    engine.reset(torch.arange(n),s)
    return engine,s


def evaluate(e,s,action=None,delayed=None,load=None,done=None,timeout=None,command=None):
    n=len(s.dof_pos); zero=lambda k:torch.zeros(n,k)
    return e.evaluate(s,zero(12) if action is None else action,zero(12) if delayed is None else delayed,
        zero(4) if command is None else command,zero(3) if load is None else load,
        torch.zeros(n,dtype=torch.bool) if done is None else done,
        torch.zeros(n,dtype=torch.bool) if timeout is None else timeout)


def raw(e,k): return e.sums['raw/'+k]/len(e.age)


def test_published_coefficients_exclusions_and_single_dt():
    weights=[2,.5,-2,-.05,-2.5e-7,-2e-5,-1e-5,-1.2,-.02,-.001,-1,-2,1,-2,2]
    assert [v[0] for v in ORIGINAL.values()]==weights
    e,s=fixture(); a=evaluate(e,s)
    torch.testing.assert_close(a,torch.full((3,),4.5*.02))
    e2,s2=fixture(dt=.04); torch.testing.assert_close(evaluate(e2,s2),a*2)
    assert not any(k in ADDITIONS for k in ('pos_action_rate','tau_action_smoothness','torque_cancellation','feedback_torques'))
    assert e.spec['terms']['dof_pos_limits']['units']=='joint count'
    assert e.spec['assumptions']['load_motion']


def test_original_formulas_limit_count_and_endpoint_torque():
    e,s=fixture(); s.base_lin_vel[:]=torch.tensor([1.,2.,3.]); s.base_ang_vel[:]=torch.tensor([2.,3.,4.])
    s.dof_pos[:,0]=2.5; s.dof_vel[:]=2; s.torques[:]=3; s.requested_torque[:]=1e4
    s.base_pos[:,2]+=.2; s.link_contact_forces[:,2,0]=1
    evaluate(e,s)
    expected=dict(tracking_lin_vel=torch.exp(torch.tensor(-20.)),tracking_ang_vel=torch.exp(torch.tensor(-64.)),
        lin_vel_z=9.,ang_vel_xy=13.,dof_acc=120000.,joint_power=72.,torques=108.,base_height=.04,collision=1.,dof_pos_limits=1.)
    for k,v in expected.items(): torch.testing.assert_close(raw(e,k),torch.as_tensor(v),rtol=1e-5,atol=1e-6)


def test_fixed_tracking_and_load_symmetry_monotonicity():
    e,s=fixture(); velocity=torch.tensor([[1.,2.,0.]]*3); omega=torch.ones(3,3); commands=torch.zeros(3,4)
    lin,yaw=tracking(velocity,omega,commands)
    s.base_lin_vel=velocity; s.base_ang_vel=omega
    evaluate(e,s,load=torch.tensor([[0.,0.,0.],[1.,2.,3.],[-1.,-2.,-3.]]))
    torch.testing.assert_close(raw(e,'tracking_lin_vel'),lin.mean()); torch.testing.assert_close(raw(e,'tracking_ang_vel'),yaw.mean())
    e2,s2=fixture(); e2.cfg.rewards.dynamic_tracking_sigma_enabled=True
    s2.base_lin_vel=velocity; s2.base_ang_vel=omega
    out=evaluate(e2,s2,load=torch.tensor([[0.,0.,0.],[1.,2.,3.],[-1.,-2.,-3.]]))
    torch.testing.assert_close(raw(e2,'tracking_lin_vel'),lin.mean())
    torch.testing.assert_close(raw(e2,'tracking_ang_vel'),yaw.mean())
    assert out[0]>out[1] and out[1]==out[2]


def test_action_histories_separate_from_delay_and_partial_reset():
    e,s=fixture(); action=torch.ones(3,12)
    evaluate(e,s,action=action,delayed=-action); assert raw(e,'action_rate')==0
    e.begin_iteration(1); evaluate(e,s,action=action*2,delayed=action); assert raw(e,'action_rate')==12
    assert raw(e,'action_smoothness')==0
    e.begin_iteration(2); evaluate(e,s,action=action*4); assert raw(e,'action_smoothness')==12
    old=e.actions[1:].clone(); e.reset(torch.tensor([0]),s)
    assert torch.equal(old,e.actions[1:]) and e.age[0]==0 and e.age[1]==3
    e.begin_iteration(3); evaluate(e,s,action=action*4)
    assert raw(e,'action_rate')==0


def test_air_time_filtered_landing_and_reset():
    e,s=fixture(); cmd=torch.ones(3,4)
    for _ in range(30): evaluate(e,s,command=cmd)
    e.begin_iteration(1); s.link_contact_forces[:,s.feet_contact_indices,2]=10
    evaluate(e,s,command=cmd); torch.testing.assert_close(raw(e,'feet_air_time'),torch.tensor(4*.12),atol=1e-6,rtol=1e-5)
    e.begin_iteration(2); evaluate(e,s,command=cmd); assert raw(e,'feet_air_time')==0
    e.reset(torch.tensor([1]),s); assert e.air[1].eq(0).all()


def test_shared_smoothness_is_zero_for_constant_actions():
    e,s=fixture(); action=torch.ones(3,12)
    evaluate(e,s,action=action); evaluate(e,s,action=action)
    e.begin_iteration(1); evaluate(e,s,action=action)
    assert raw(e,'action_smoothness')==0  # (1-2*1+1)^2 * 12


def test_joint_limit_count_uses_strict_threshold_not_excess():
    e,s=fixture()
    s.dof_pos[:,0]=s.dof_pos_limits[0,0]; s.dof_pos[:,1]=s.dof_pos_limits[1,1]
    evaluate(e,s); assert raw(e,'dof_pos_limits')==0
    e.begin_iteration(1); s.dof_pos[:,0]-=1e-4; s.dof_pos[:,1]+=100
    evaluate(e,s); assert raw(e,'dof_pos_limits')==2


def test_overlapping_definitions_match_hardpact_helpers():
    from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
    e,s=fixture(); gen=torch.Generator().manual_seed(7)
    s.dof_pos=torch.randn(3,12,generator=gen)*2
    s.dof_vel=torch.randn(3,12,generator=gen)
    s.torques=torch.randn(3,12,generator=gen)*10
    s.link_contact_forces=torch.randn(3,7,3,generator=gen)*500
    action=torch.randn(3,12,generator=gen)
    e.actions.copy_(torch.randn(3,2,12,generator=gen)); e.age[:]=3
    e.air[:]=.6
    commands=torch.ones(3,4)*.2
    reference=deepcopy(s); reference.last_dof_vel=e.velocity.clone()
    reference.penalized_contact_indices=s.penalised_contact_indices
    ctx=NS(simulator=reference,cfg=e.cfg,commands=commands,actions=action,dt=e.dt,
        last_actions=e.actions[:,0].clone(),llast_actions=e.actions[:,1].clone(),
        feet_air_time=e.air.clone(),last_contacts=e.contacts.clone())
    excluded={'load_motion','tracking_lin_vel','tracking_ang_vel','dof_pos_limits'}
    expected={key:getattr(Go2PACT,'_reward_'+key)(ctx).mean() for key in ORIGINAL if key not in excluded}
    evaluate(e,s,action=action,command=commands)
    for key,value in expected.items(): torch.testing.assert_close(raw(e,key),value)


def test_tracking_ignores_terrain_wrench_sigma_and_matches_fixed_formula():
    from legged_gym.envs.go2.go2_hard_pact.domain_rand_curriculum import HardPACTDomainRandCurriculum
    e,s=fixture(); s.terrain_levels=torch.full((3,),9); s.terrain_kind_ids=torch.full((3,),4)
    commands=torch.tensor([[1.,1.,2.,0.]]*3)
    tc=NS(domain_rand_curriculum=HardPACTDomainRandCurriculum(e.cfg),
        _current_sustained_wrench_world=torch.full((3,6),1000.))
    def run():
        e.evaluate(s,torch.zeros(3,12),torch.zeros(3,12),commands,torch.zeros(3,3),
            torch.zeros(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool),tracking_context=tc)
    run()
    raw_linear,raw_yaw=tracking(s.base_lin_vel,s.base_ang_vel,commands)
    torch.testing.assert_close(raw(e,'tracking_lin_vel'),raw_linear.mean())
    torch.testing.assert_close(raw(e,'tracking_ang_vel'),raw_yaw.mean())
    relaxed=raw(e,'tracking_lin_vel').clone()
    e.begin_iteration(1); tc._current_sustained_wrench_world.zero_(); run()
    torch.testing.assert_close(raw(e,'tracking_lin_vel'),relaxed)
    e.begin_iteration(2); e.cfg.rewards.dynamic_tracking_sigma_enabled=False; run()
    torch.testing.assert_close(raw(e,'tracking_lin_vel'),raw_linear.mean())
    torch.testing.assert_close(raw(e,'tracking_ang_vel'),raw_yaw.mean())


def test_old_literal_reward_checkpoint_rejected():
    e,_=fixture(); state=e.state_dict(); state['version']=1
    with pytest.raises(ValueError,match='reward contract'): e.load_state_dict(state)


def test_shaping_terminal_timeout_and_reset_potentials():
    e,s=fixture(additions=True); s.base_pos[:,2]+=.3
    done=torch.tensor([True,True,False]); timeout=torch.tensor([False,True,False])
    old=e.potential.clone(); evaluate(e,s,done=done,timeout=timeout)
    p=e.potentials(s); expected=p[:,1]-old[:,1]; expected[0]=-old[0,1]
    torch.testing.assert_close(e.sums['raw/pbrs_height'],expected.sum())
    e.reset(torch.tensor([0]),s); torch.testing.assert_close(e.potential[0],p[0])


def test_shared_slip_vhip_and_unclipped_penalty():
    from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
    e,s=fixture(additions=True); s.feet_vel[:,:,:2]=torch.tensor([1.,-1.])
    s.link_contact_forces[:,s.feet_contact_indices,2]=10
    s.requested_torque[:]=30; evaluate(e,s)
    assert raw(e,'foot_slip')==0  # Shared square(sum(vxy)): opposite components cancel.
    assert raw(e,'torque_limits')==12*(30-20*.9)
    e.begin_iteration(1)
    s.base_pos[:,2]+=10; s.feet_pos[:,:,2]+=10; s.measured_heights+=10
    evaluate(e,s)
    for key in ('foot_slip','vhip_angle','vhip_angular_acc'):
        torch.testing.assert_close(raw(e,key),getattr(Go2PACT,'_reward_'+key)(e._shared_context).mean())


def test_curriculum_protection_endpoints_and_resume():
    e,s=fixture(additions=True); schedule=e.spec['curriculum']; schedule.update(warmup=2,duration=4)
    e.begin_iteration(0); assert e.coefficients()['orientation']==-.2
    assert e.coefficients()['lin_vel_z']==-2.
    e.begin_iteration(4); assert e.coefficients()['orientation']==pytest.approx(-.6)
    state=e.state_dict(); e2,_=fixture(additions=True); e2.spec=deepcopy(e.spec); e2.load_state_dict(state)
    assert e.coefficients()==e2.coefficients()
    e.begin_iteration(6); assert e.coefficients()['orientation']==-1.
    with pytest.raises(ValueError,match='protected'):
        resolve_beyond_robustness(reward_overrides={'curriculum':{'bounds':{'lin_vel_z':[-1,-2]}}})


def test_merged_spec_isolation_and_smoke_opt_in():
    a=resolve_beyond_robustness(); b=resolve_beyond_robustness()
    assert a.environment.br_reward['mode']=='merged'
    a.environment.br_reward['terms']['orientation']['coefficient']=123
    assert b.environment.br_reward['terms']['orientation']['coefficient']!=123
    assert resolve_beyond_robustness(reward_overrides={'mode':'smoke_tracking'}).environment.br_reward['mode']=='smoke_tracking'
