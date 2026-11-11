from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch

from rsl_rl.algorithms.hard_pact_actor_regularizers import (
    ActorRegularizerConfig, fk_consistency_loss, compensation_demand, load_compensation_loss,
)
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT


def allocation_inputs():
    J=torch.zeros(3,4,3,18,dtype=torch.float64)
    J[:,:,:,:3]=torch.eye(3,dtype=J.dtype)
    # Point-contact world Jacobians: angular velocity cross foot position.
    positions=torch.tensor([[.3,-.2,-.4],[.3,.2,-.4],[-.3,-.2,-.4],[-.3,.2,-.4]],dtype=J.dtype)
    for i,(x,y,z) in enumerate(positions):
        J[:,i,:,3:6]=torch.tensor([[0,z,-y],[-z,0,x],[y,-x,0]],dtype=J.dtype)
        J[:,i,:,6+3*i:9+3*i]=torch.eye(3,dtype=J.dtype)
    B=torch.zeros(3,6,18,dtype=J.dtype); B[:,:,:6]=torch.eye(6,dtype=J.dtype)
    W=torch.zeros(3,6,dtype=J.dtype)
    gravity=torch.zeros(3,18,dtype=J.dtype);gravity[:,2]=80
    stance=torch.ones(3,4,dtype=torch.bool)
    return J,B,W,gravity,stance


def test_exact_support_gravity_external_load_and_asymmetry():
    cfg=ActorRegularizerConfig();J,B,W,g,stance=allocation_inputs()
    W[1,2]=-40
    g[2,4]=4  # shifted support center: asymmetric front/rear loading
    for x in (J,B,W,g):x.requires_grad_()
    target,valid=compensation_demand(J,B,W,g,stance,cfg,.6)
    assert valid.all() and not target.requires_grad
    torch.testing.assert_close(target[0,2::3],torch.full((4,),-20.,dtype=g.dtype))
    torch.testing.assert_close(target[1,2::3],torch.full((4,),-30.,dtype=g.dtype))
    forces=-target.reshape(3,4,3)
    torch.testing.assert_close(torch.einsum('bfkn,bfk->bn',J.detach(),forces)[:,:6],
        (g.detach()-torch.einsum('bkn,bk->bn',B.detach(),W.detach()))[:,:6],atol=1e-10,rtol=1e-10)
    assert not torch.allclose(target[2,2::3],target[0,2::3])


def test_two_contact_rank_deficient_achievable_and_infeasible():
    J,B,W,g,stance=allocation_inputs()
    stance[:]=False;stance[:,[0,3]]=True  # diagonal support through base
    g[1,3:6]=torch.tensor([.6,.4,0.],dtype=g.dtype)  # moment along support line: unattainable
    stance[2]=False
    target,valid=compensation_demand(J,B,W,g,stance,ActorRegularizerConfig(),.6)
    assert valid.tolist()==[True,False,False]
    torch.testing.assert_close(target[0,[2,11]],torch.full((2,),-40.,dtype=g.dtype))
    assert target[1:].eq(0).all()


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),1e300])
def test_support_invalid_inputs_do_not_poison_neighbors(bad):
    J,B,W,g,stance=allocation_inputs()
    J[1]=bad
    target,valid=compensation_demand(J,B,W,g,stance,ActorRegularizerConfig(),.6)
    assert valid.tolist()==[True,False,True] and target.isfinite().all()


def test_support_friction_rejection_and_no_stale_target():
    J,B,W,g,stance=allocation_inputs();cfg=ActorRegularizerConfig()
    original,valid=compensation_demand(J,B,W,g,stance,cfg,.6)
    W[:,2]=-40
    changed,_=compensation_demand(J,B,W,g,stance,cfg,.6)
    assert not torch.equal(original,changed)
    W[:,0]=10000
    assert not compensation_demand(J,B,W,g,stance,cfg,.6)[1].any()
    W.zero_();g[:,2]=-80
    assert not compensation_demand(J,B,W,g,stance,cfg,.6)[1].any()


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA unavailable'))])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_full_and_rank_deficient_minimum_norm_parity(device,dtype):
    J,B,W,g,stance=[x.to(device) for x in allocation_inputs()]
    J,B,W,g=[x.to(dtype) for x in (J,B,W,g)]
    stance[1]=False;stance[1,[0,3]]=True
    diagnostics={}
    target,valid=compensation_demand(J,B,W,g,stance,ActorRegularizerConfig(),.6,diagnostics)
    C=(J*stance[:,:,None,None]).flatten(1,2)[:,:,:6].transpose(1,2)
    expected=torch.linalg.pinv(C.double())@g[:,:6].double()[...,None]
    torch.testing.assert_close(-target.reshape(3,12).double(),expected.squeeze(-1),
                              atol=2e-4 if dtype==torch.float32 else 1e-10,rtol=1e-5)
    assert valid.all() and diagnostics['rank_deficient_fraction']>0


def test_direct_huber_valid_stance_coordinates_and_gradients():
    cfg=ActorRegularizerConfig();J,B,W,g,stance=allocation_inputs()
    target,feasible=compensation_demand(J,B,W,g,stance,cfg,.6)
    ff=torch.zeros(3,12,dtype=g.dtype,requires_grad=True)
    with torch.no_grad():ff[1]=float('nan')
    valid=torch.tensor([True,False,False])
    loss,metrics=load_compensation_loss(ff,target,torch.ones(12)*20,stance,valid,feasible,cfg)
    torch.testing.assert_close(loss,torch.tensor(1/6,dtype=g.dtype))
    loss.backward()
    assert ff.grad.isfinite().all() and ff.grad[1:].eq(0).all() and ff.grad[0,2::3].gt(0).all()
    assert metrics['valid_joint_count']==12
    good,_=load_compensation_loss(target,target,torch.ones(12)*20,stance,valid,feasible,cfg)
    assert good==0
    empty,_=load_compensation_loss(ff,target,torch.ones(12)*20,stance,valid&False,feasible,cfg)
    assert empty.isfinite() and empty==0


def test_fk_masks_deadband_and_target_detachment():
    p=torch.ones(2,4,3,requires_grad=True);target=torch.zeros_like(p,requires_grad=True)
    with torch.no_grad(): target[1]=float('nan')
    loss,metrics=fk_consistency_loss(p,target,torch.ones(2,4,dtype=torch.bool),
        torch.tensor([True,False]),ActorRegularizerConfig())
    loss.backward();assert p.grad[0].abs().sum()>0 and p.grad[1].eq(0).all() and target.grad is None
    zero,_=fk_consistency_loss(p,target,torch.ones(2,4,dtype=torch.bool),torch.tensor([True,False]),
        ActorRegularizerConfig(fk_deadband_m=1.1))
    assert zero==0 and metrics['valid_fraction']==.5


def test_channel_vjp_isolates_ff_from_position_estimator_and_shared_context():
    pos=torch.nn.Parameter(torch.tensor(2.));ff=torch.nn.Parameter(torch.tensor(3.))
    trunk=torch.nn.Parameter(torch.tensor(4.));head=torch.nn.Parameter(torch.tensor(5.))
    q=pos*trunk;t=ff*trunk;nominal=q+t
    candidate=nominal*2+head*trunk
    loss=candidate.square()
    routed=PPO_HardPACT._actor_only_torque_vjp(loss,nominal,[pos,ff],branch=t)
    routed.backward()
    assert ff.grad!=0 and pos.grad is None and trunk.grad is None and head.grad is None


def test_support_nominal_ff_actor_only_routing_without_qp():
    J,B,W,g,stance=allocation_inputs()
    cfg=ActorRegularizerConfig()
    target,feasible=compensation_demand(J,B,W,g,stance,cfg,.6)
    pos=torch.nn.Parameter(torch.ones(3,12,dtype=g.dtype))
    ff=torch.nn.Parameter(torch.full_like(pos,.1));context=torch.nn.Parameter(torch.ones_like(pos))
    trunk=torch.nn.Parameter(torch.ones_like(pos))
    command=ff*trunk*context
    loss,_=load_compensation_loss(.8*command,target,torch.ones(12)*20,stance,
        torch.ones(3,dtype=torch.bool),feasible,cfg)
    routed=PPO_HardPACT._actor_only_torque_vjp(loss,command,[pos,ff,trunk])
    routed.backward()
    assert ff.grad.abs().sum()>0 and ff.grad.isfinite().all()
    assert trunk.grad.abs().sum()>0
    assert pos.grad is None and context.grad is None


def test_nominal_gravity_is_not_velocity_bias():
    from legged_gym.dynamics import BardGo2Dynamics
    from rsl_rl.algorithms.hard_pact_actor_regularizers import nominal_support_gravity
    dynamics=BardGo2Dynamics('resources/robots/go2/urdf/go2.urdf',dtype=torch.float64,batch_capacity=2)
    q=torch.zeros(1,19,dtype=torch.float64);q[:,3]=1
    q[:,7:]=torch.tensor([.1,.8,-1.5]*4)
    gravity=nominal_support_gravity(dynamics,q)
    context=dynamics.build_context(q,torch.zeros(1,18,dtype=q.dtype),parameters={},need_jacobians=False)
    torch.testing.assert_close(gravity,context.rnea(torch.zeros(1,18,dtype=q.dtype)))
    context=dynamics.build_context(q,torch.ones(1,18,dtype=q.dtype),parameters={},need_jacobians=False)
    assert not torch.allclose(gravity,context.rnea(torch.zeros(1,18,dtype=q.dtype)))
    assert not gravity.requires_grad


def test_bard_fk_matches_finite_difference_in_canonical_order():
    from legged_gym.dynamics import BardGo2Dynamics
    dynamics=BardGo2Dynamics('resources/robots/go2/urdf/go2.urdf',dtype=torch.float64,batch_capacity=2)
    q=torch.tensor([[.1,.8,-1.5]*4],dtype=torch.float64,requires_grad=True)
    p=dynamics.foot_positions_body(q);seed=torch.arange(12,dtype=q.dtype).reshape(1,4,3)/12
    gradient=torch.autograd.grad((p*seed).sum(),q)[0]
    numeric=torch.zeros_like(q);eps=1e-6
    for j in range(12):
        perturb=torch.zeros_like(q);perturb[0,j]=eps
        numeric[0,j]=((dynamics.foot_positions_body(q.detach()+perturb)-dynamics.foot_positions_body(q.detach()-perturb))*seed).sum()/(2*eps)
    torch.testing.assert_close(gradient,numeric,atol=1e-7,rtol=1e-5)


def test_fk_uses_delayed_command_and_filters_invalid_rows_before_fk():
    alg=PPO_HardPACT.__new__(PPO_HardPACT)
    alg.actor_regularizers=ActorRegularizerConfig(lambda_fk=1.)
    parameter=torch.nn.Parameter(torch.ones(3,12))
    alg.ppo_parameters=[parameter];alg._actor_regularizer_diagnostics_due=False
    alg.physics_dynamics=SimpleNamespace(foot_positions_body=Mock(side_effect=lambda q:q.reshape(-1,4,3)))
    command=parameter*2
    batch={'physics_valid_mask':torch.tensor([[1],[0],[1]],dtype=torch.bool),
        'delayed_action_source_valid':torch.tensor([[1],[1],[0]],dtype=torch.bool),
        'actor_fk_stance':torch.ones(3,4,dtype=torch.bool),'actor_fk_next_feet_body':torch.zeros(3,4,3)}
    batch['actor_fk_next_feet_body'][1]=float('nan')
    loss=alg._compute_actor_fk_loss({'desired_position':command,'current_action':parameter*99},batch)
    torch.testing.assert_close(alg.physics_dynamics.foot_positions_body.call_args.args[0],command[:1])
    loss.backward();assert parameter.grad[0].abs().sum()>0 and parameter.grad[1:].eq(0).all()


def test_disabled_defaults_and_config_validation():
    cfg=ActorRegularizerConfig()
    assert cfg.lambda_fk==cfg.lambda_load==0
    with pytest.raises(ValueError): ActorRegularizerConfig(fk_position_scale_m=0)
    with pytest.raises(ValueError): ActorRegularizerConfig(lambda_load=float('nan'))


def test_target_capture_pre_reset_body_frame_and_continuous_stance():
    from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
    env=Go2HardPACT.__new__(Go2HardPACT);env._actor_fk_enabled=True
    quat=torch.tensor([[0.,0.,2**-.5,2**-.5]])
    env._current_base_quat_xyzw=lambda:quat
    env._canonical_configuration=lambda:torch.zeros(1,19)
    env._canonical_velocity_world=lambda:torch.zeros(1,18)
    env.simulator=SimpleNamespace(base_pos=torch.tensor([[2.,3.,4.]]),
        feet_pos=torch.tensor([[[2.,4.,4.]]*4]),hard_pact_foot_forces_world=lambda:torch.zeros(1,4,3))
    env.grf_processor=SimpleNamespace(contacts=torch.ones(1,4,dtype=torch.bool),update_substep=lambda *a,**k:None)
    env._capture_bard_pre_state()
    env.grf_processor.contacts[:,1]=False;env._hard_pact_grf_post_physics_substep()
    env.grf_processor.contacts[:]=True;env._hard_pact_grf_post_physics_substep()
    env._capture_bard_post_state()
    torch.testing.assert_close(env._actor_fk_next_feet_body,torch.tensor([[[1.,0.,0.]]*4]),atol=2e-7,rtol=0)
    assert env._actor_fk_stance.tolist()==[[True,False,True,True]]
    env.simulator.feet_pos.zero_()  # reset cannot mutate captured supervision
    assert env._actor_fk_next_feet_body[:,:,0].min()>.99


@pytest.mark.skipif(not torch.cuda.is_available(),reason='cuPIQP needs CUDA')
@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('diagnostics',[False,True])
def test_tiny_real_ppo_update_reuses_one_qp_and_disabled_paths_are_not_called(monkeypatch,enabled,diagnostics):
    import test_hard_pact_contact_indexing_and_inverse_gate as live
    import rsl_rl.algorithms.ppo_hard_pact as module
    factory=live.make_algorithm;algorithms=[];calls=[]
    def make(**kwargs):
        alg=factory(**kwargs,actor_regularizers=dict(lambda_fk=.01 if enabled else 0.,lambda_load=.01 if enabled else 0.))
        algorithms.append(alg);return alg
    monkeypatch.setattr(live,'make_algorithm',make)
    update=PPO_HardPACT.update
    def prepare(self,*args,**kwargs):
        self.qp_config=replace(self.qp_config,tensorboard_diagnostics_enabled=diagnostics)
        if enabled:
            f=self.storage.hard_pact_fields
            f['physics_valid_mask']=~(f['reset_mask']|f['timeout_mask']|f['push_event_mask']|f['teleport_mask'])
            f['actor_fk_stance']=torch.ones(2,4,4,device=self.device,dtype=torch.bool)
            f['actor_fk_next_feet_body']=torch.zeros(2,4,4,3,device=self.device)
        return update(self,*args,**kwargs)
    monkeypatch.setattr(PPO_HardPACT,'update',prepare)
    load=module.compensation_demand;fk=PPO_HardPACT._compute_actor_fk_loss
    def checked_load(*a,**k):
        assert enabled;calls.append('load');return load(*a,**k)
    def checked_fk(*a,**k):
        assert enabled;calls.append('fk');return fk(*a,**k)
    monkeypatch.setattr(module,'compensation_demand',checked_load)
    monkeypatch.setattr(PPO_HardPACT,'_compute_actor_fk_loss',checked_fk)
    with torch.device('cuda'):live._run_owned_update('random_one_substep','cuda')
    assert calls==(['load','fk'] if enabled else [])
    metrics=algorithms[0].last_actor_regularizer_metrics
    assert bool(metrics)==(enabled and diagnostics)
    assert all(torch.isfinite(value) for value in metrics.values())
    losses=algorithms[0].last_actor_loss_metrics
    names={'projection_primary','projection_recovery','velocity_xy','velocity_yaw',
           'fk','load','total_weighted','total_warmed',
           'projection_primary/torque_correction_mae_nm',
           'projection_primary/stance_acceleration_error_m_s2',
           'projection_recovery/torque_correction_mae_nm',
           'projection_recovery/joint_slack_rad_s2','projection_recovery/rate_slack_nm',
           'velocity_xy/error_m_s','velocity_yaw/error_rad_s'}
    if enabled:
        names.update(('fk/error_m','load/error_nm'))
    assert set(losses)=={'Loss/actor_pinn/'+name for name in names}
    assert all(torch.isfinite(value) and not value.requires_grad for value in losses.values())
    assert losses['Loss/actor_pinn/fk']>0 if enabled else losses['Loss/actor_pinn/fk']==0
    torch.testing.assert_close(losses['Loss/actor_pinn/total_weighted'],losses['Loss/actor_pinn/total_warmed'])


def test_actor_losses_log_off_cadence_with_valid_count_weighting_and_no_graph():
    alg=PPO_HardPACT.__new__(PPO_HardPACT)
    alg._actor_regularizer_diagnostics_due=False
    alg._actor_loss_sums={}
    value=torch.tensor(2.,requires_grad=True)
    alg._record_actor_regularizer('fk',{'loss':value},1,4)
    alg._record_actor_regularizer('fk',{'loss':value*3},3,4)
    alg._record_actor_regularizer('fk',{'loss':value*float('nan')},0,4)
    alg._finish_actor_loss_logging(torch.tensor(0.))
    assert alg.last_actor_loss_metrics['Loss/actor_pinn/fk']==5
    assert alg.last_actor_loss_metrics['Loss/actor_pinn/load']==0
    assert all(not v.requires_grad for v in alg.last_actor_loss_metrics.values())
    assert value.grad is None
    alg._actor_loss_sums={}
    alg._finish_actor_loss_logging(torch.tensor(0.))
    assert all(v==0 for v in alg.last_actor_loss_metrics.values())
