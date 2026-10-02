"""Height-aware explicit estimator, affine QP regulation and torso scheduling."""
from dataclasses import replace
import pytest
import torch
from test_hard_pact_reduced_qp import solver, inputs
from rsl_rl.modules.hard_pact_physics import (
    terrain_relative_torso_height, compose_explicit_estimator_target,
    estimated_qp_height_inputs, ExplicitEstimatorDecoder)
from rsl_rl.algorithms.hard_pact_qp_curriculum import QPCurriculum


def test_terrain_label_and_predicted_velocity_rotation():
    base=torch.tensor([[0.,0.,1.],[0.,0.,2.]])
    terrain=torch.tensor([[.1,.2,.3],[1.,1.,1.]])
    h=terrain_relative_torso_height(base,terrain)
    torch.testing.assert_close(h,torch.tensor([[.8],[1.]]))
    target=compose_explicit_estimator_target(torch.tensor([[2.,0.,0.],[0.,0.,2.]]),torch.ones(2,4),torch.zeros(2,4),h)
    assert target.shape==(2,12)
    q=torch.tensor([[0.,2**-.5,0.,2**-.5],[0.,0.,0.,1.]])
    target.requires_grad_();q.requires_grad_()
    state=estimated_qp_height_inputs(target,q,2.)
    torch.testing.assert_close(state['estimated_base_linear_velocity_world'],torch.tensor([[0.,0.,-1.],[0.,0.,1.]]),atol=2e-7,rtol=0)
    assert all(not v.requires_grad for v in state.values())


def height_data():
    d=inputs(2)
    q=d['base_quaternion'];q[:]=torch.tensor([0.,2**-.5,0.,2**-.5])
    e=torch.zeros(2,12,dtype=torch.float64);e[:,:3]=torch.tensor([.6,.2,-.4]);e[:,11]=.2
    d.update(estimated_qp_height_inputs(e,q,2.))
    R=torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]],dtype=torch.float64)
    d['base_jacobian'][:,:3,:3]=R;d['base_jacobian'][:,3:6,3:6]=R
    d['foot_jacobians'][:,:,:,:3]=R
    d['mass_matrix'][:,0,6]=d['mass_matrix'][:,6,0]=.1
    d['base_angular_velocity_world'][:,1]=1.
    return d


@pytest.mark.parametrize('backend',['qpth','cupiqp'])
def test_height_quadratic_bias_recovery_and_detached_state(backend):
    if backend=='cupiqp' and not torch.cuda.is_available():pytest.skip('CUDA required')
    d=height_data()
    if backend=='cupiqp':d={k:v.cuda() for k,v in d.items()}
    q=solver(height_weight=0.,height_target=.38,qp_solver=backend,solver_dtype='float64')
    before=q._build(d)
    # Zero weight does not even inspect height inputs.
    absent={k:v for k,v in d.items() if not k.startswith('estimated_')}
    assert torch.equal(q._build(absent).Q,before.Q)
    q.cfg=replace(q.cfg,height_weight=3.)
    d['estimated_height'].requires_grad_();d['estimated_base_linear_velocity_world'].requires_grad_()
    d['tau_nom'].requires_grad_();d['wrench_pred_world'].requires_grad_()
    m=q._build(d)
    C,e,desired=q._height_affine(d,m.acceleration_map,m.acceleration_offset)
    expected_desired=20*(.38-d['estimated_height'].flatten())-5*d['estimated_base_linear_velocity_world'][:,2]
    torch.testing.assert_close(desired,expected_desired)
    # omega_y=1: (omega cross v)_z=-vx, which is +0.2 here.
    torch.testing.assert_close(e.flatten()+desired-(d['base_jacobian'][:,2:3]@m.acceleration_offset[...,None]).flatten(),e.new_full((2,),.2),atol=1e-7,rtol=1e-6)
    Cz=C*m.variable_scale/q.cfg.height_acceleration_scale;ez=e/q.cfg.height_acceleration_scale
    torch.testing.assert_close(m.Q-before.Q,6*Cz.transpose(1,2)@Cz,atol=1e-12,rtol=1e-8)
    torch.testing.assert_close(m.p-before.p,6*(Cz.transpose(1,2)@ez[...,None]).squeeze(-1))
    assert torch.equal(m.G,before.G) and torch.equal(m.h,before.h)
    assert torch.equal(q._soft_joint_problem(m).Q[:,:24,:24],m.Q)
    out=q.solve(differentiable=True,**d)
    grads=torch.autograd.grad(out.tau_safe.sum(),(d['tau_nom'],d['wrench_pred_world'],d['estimated_height'],d['estimated_base_linear_velocity_world']),allow_unused=True)
    assert torch.isfinite(grads[0]).all() and grads[0].abs().sum()>0
    assert torch.isfinite(grads[1]).all() and grads[2:] == (None,None)


def test_estimator_supervision_and_explicit_legacy_rejection():
    from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
    from rsl_rl.algorithms.ppo_pact_pos import PPO_PACT_Pos
    decoder=ExplicitEstimatorDecoder(8,(8,),12)
    out=decoder(torch.ones(2,8));target=out.explicit_for_policy.detach().clone();target[:,11]+=1
    valid=torch.ones(2,dtype=torch.bool)
    for loss in (PPO_HardPACT._masked_explicit_loss,PPO_PACT_Pos._masked_explicit_loss):
        value=loss(out.explicit_for_policy,out.contact_logits,target,valid)
        grad=torch.autograd.grad(value,decoder.network[-1].weight,retain_graph=True)[0]
        assert grad[11].abs().sum()>0
    with pytest.raises(RuntimeError,match='11-D legacy'):
        decoder.load_state_dict(ExplicitEstimatorDecoder(8,(8,),11).state_dict())


def test_physical_height_metrics_and_truth_is_diagnostic_only():
    d=height_data();q=solver(height_weight=1.,height_target=.38,diagnostics_level='physical')
    d['diagnostic_height_truth']=d['estimated_height']+.1
    out=q.solve(differentiable=False,**d)
    metrics=q.iteration_metrics('rollout',d['tau_nom'])
    p='qp/rollout/model_height/primary/accepted/'
    assert metrics[p+'height_estimation_abs_error_m']==pytest.approx(.1)
    assert metrics[p+'height_error_m']==pytest.approx(.18)
    assert metrics[p+'weighted_cost']>=0
    assert all(not value.requires_grad for name,value in metrics.items() if name.startswith(p))
    # Truth cannot affect assembly/acceptance, even if diagnostic-only truth is invalid.
    d['diagnostic_height_truth'].fill_(float('nan'))
    other=q.solve(differentiable=False,**d)
    torch.testing.assert_close(out.tau_safe,other.tau_safe,atol=0,rtol=0)
    assert torch.equal(out.stage,other.stage)


def test_authoritative_endpoints_and_resumed_torso_schedule():
    q=solver().cfg
    cfg=replace(q,objective_curriculum_enabled=True,height_weight=4.,attitude_weight=2.,
        contact_acceleration_weight=8.,attitude_weight_final=999.,contact_acceleration_weight_final=999.,
        objective_curriculum_baseline_override=.8,objective_curriculum_ema_alpha=1.,
        objective_curriculum_recovery_iterations=1,objective_curriculum_step_interval=1)
    s=QPCurriculum(cfg);start,_=s.begin(0)
    assert (start.height_weight,start.attitude_weight,start.contact_acceleration_weight)==(1.,.5,2.)
    s.finish(0,1.,10)
    restored=QPCurriculum(cfg);restored.load_state_dict(s.state_dict())
    assert s.begin(1)==restored.begin(1)
    s.progress=1.;s.snapshot_iteration=None;end,_=s.begin(2)
    assert (end.height_weight,end.attitude_weight,end.contact_acceleration_weight)==(4.,2.,8.)
    direct,_=QPCurriculum(replace(cfg,objective_curriculum_enabled=False)).begin(0)
    assert direct.height_weight==4. and direct.attitude_weight==2.
    partial,_=QPCurriculum(replace(cfg,torso_stability_curriculum_enabled=False)).begin(0)
    assert partial.height_weight==4. and partial.contact_acceleration_weight==2.


@pytest.mark.parametrize('mode',['every_substep','random_one_substep'])
def test_rollout_inputs_reconstruct_from_same_explicit_and_sampled_pose(mode):
    from test_hard_pact_qp_modes import fixture
    task,_,qp,_,_,q,v,quat,_=fixture(mode)
    qp.cfg=replace(qp.cfg,height_weight=1.,height_target=.38,height_velocity_obs_scale=2.)
    task._hard_pact_policy_explicit=torch.cat((task._hard_pact_policy_explicit,torch.full((8,1),.3)),1)
    original=qp.solve;seen=[]
    def solve(**kwargs):
        expected=estimated_qp_height_inputs(task._hard_pact_policy_explicit[kwargs['environment_ids']],kwargs['base_quaternion'],2.)
        for key in expected:torch.testing.assert_close(kwargs[key],expected[key])
        seen.append(kwargs['environment_ids'].numel())
        return original(**kwargs)
    qp.solve=solve
    task._begin_qp_interval()
    for _ in range(4):task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(8,6))
    assert sum(seen)==(32 if mode=='every_substep' else 8)


def test_height_aware_pos_migration_and_contract():
    from rsl_rl.modules import ActorCritic_HardPACT,ActorCritic_HardPACT_Pos
    from rsl_rl.runners.pact_pos_runner import build_hard_pact_start_checkpoint
    from legged_gym.envs.go2.go2_hard_pact.deployment import calculate_physics_head_gains,build_deployment_contract
    from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg,GO2HardPACTCfgPPO
    from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import GO2HardPACTPosCfgPPO
    cfg=GO2HardPACTCfg();gains=calculate_physics_head_gains(cfg)
    kw=dict(num_actor_obs=57,num_critic_obs=64,num_actions=12,actor_layers=[16,16],critic_layers=[16,16],
        cenet_in_dim=570,cenet_latent_dim=16,cenet_velo_dim=12,cenet_enc_layers=[16,16],
        grf_scale_n=gains.grf_scale_n,wrench_scale=gains.wrench_scale_n_nm,wrench_qp_clip=gains.wrench_qp_clip_n_nm)
    pos=ActorCritic_HardPACT_Pos(**kw);full=ActorCritic_HardPACT(**kw)
    ckpt=build_hard_pact_start_checkpoint(pos.state_dict(),{},1)
    full.load_state_dict(ckpt['model_state_dict'],strict=True)
    features=torch.ones(2,pos.context_encoder.feature_dim)
    torch.testing.assert_close(pos.explicit_estimator(features).explicit_for_policy,full.explicit_estimator(features).explicit_for_policy)
    contract=build_deployment_contract(cfg,full,gains)
    assert contract['explicit_estimator']['dimension']==12
    assert contract['explicit_estimator']['fields'][-1]['units']=='m'
    assert GO2HardPACTCfgPPO.policy.cenet_velo_dim==GO2HardPACTPosCfgPPO.policy.cenet_velo_dim==12
