from dataclasses import replace
from copy import deepcopy
import pytest
import torch
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig, qp_interval_selection
from rsl_rl.algorithms.hard_pact_qp_curriculum import (
    execution_torque, extra_intervention_strength, blended_candidate_acceleration)
from rsl_rl.algorithms.hard_pact_intervention_curriculum import legacy_intervention_contract, InterventionCurriculum
from test_hard_pact_qp_modes import fixture


@pytest.mark.parametrize('n',[0,1,7,4096])
def test_selection_rng_and_baseline_parity(n):
    g=torch.Generator().manual_seed(31); h=torch.Generator().manual_seed(31)
    args=dict(intervention_probabilities=(.5,.3,.2))
    old=qp_interval_selection('curriculum_substeps',n,4,'cpu',generator=g,**args)
    selected,anchor,baseline=qp_interval_selection('curriculum_substeps',n,4,'cpu',generator=h,
                                                   return_baseline=True,**args)
    assert torch.equal(g.get_state(),h.get_state())
    assert torch.equal(old[0],selected) and torch.equal(old[1],anchor)
    assert ((baseline & selected)==baseline).all() and (baseline & 1).all()
    assert torch.stack([(baseline & (1<<k))!=0 for k in range(4)]).sum(0).eq(2).all()


@pytest.mark.parametrize('start',[float('nan'),float('inf'),-.1,1.1])
def test_bad_strength(start):
    with pytest.raises(ValueError): HardPACTQPConfig(intervention_extra_strength_start=start)


def test_schedule_endpoints_frozen_resume_and_disabled_legacy():
    cfg=HardPACTQPConfig(qp_update_mode='curriculum_substeps',intervention_extra_strength_enabled=True)
    s=InterventionCurriculum(cfg); s.progress=.4; s.begin(5)
    assert extra_intervention_strength(cfg,0)==.25 and extra_intervention_strength(cfg,1)==1
    assert extra_intervention_strength(replace(cfg,intervention_extra_strength_enabled=False),.4)==1
    assert extra_intervention_strength(replace(cfg,qp_update_mode='every_substep'),.4)==1
    s.progress=.9
    assert extra_intervention_strength(cfg,s.snapshot_progress)==pytest.approx(.55)
    r=InterventionCurriculum(cfg); r.load_state_dict(deepcopy(s.state_dict()))
    assert r.snapshot_progress==s.snapshot_progress and r.start==s.start
    old={'qp_update_mode':'curriculum_substeps'}
    migrated=legacy_intervention_contract(old,vars(cfg))
    assert migrated['intervention_extra_strength_enabled'] is False
    assert HardPACTQPConfig.from_dict(old).intervention_extra_strength_enabled is False


@pytest.mark.parametrize('alpha',[0.,.25,1.])
def test_primary_recovery_fallback_and_exact_scalar_parity(alpha):
    base=torch.randn(3,12); candidate=torch.randn(3,12)
    accepted=torch.tensor([True,True,False])
    scalar=execution_torque(base,candidate,accepted,alpha)
    rows=execution_torque(base,candidate,accepted,torch.full((3,1),alpha))
    torch.testing.assert_close(rows,scalar,rtol=0,atol=0)
    assert torch.equal(rows[-1],candidate[-1])


def test_control_anchor_strength_and_no_extra_solves():
    task,heads,qp,_,counts,q,v,quat,_=fixture('curriculum_substeps',n=10)
    qp.cfg=replace(qp.cfg,intervention_extra_strength_enabled=True)
    task._qp_intervention_probabilities=(.5,.3,.2)
    task._qp_extra_strength=.4; task._qp_execution_alpha=.8
    task._begin_qp_interval()
    anchors=task._qp_sampled_substep_index.clone()
    baseline=(task._qp_baseline_selection & (1<<anchors))!=0
    calls=heads.calls
    for k in range(4):
        q.fill_(.001*k);v.fill_(.0001*k)
        task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(10,6))
    assert counts.sum()==27 and calls==heads.calls
    expected=torch.where(baseline,.8,.32)
    torch.testing.assert_close(task._qp_sampled_transition['sampled_qp_execution_alpha'][:,0],expected)
    # Owned scalar replay fields survive the same indexed minibatch operation.
    order=torch.randperm(10)
    torch.testing.assert_close(task._qp_sampled_transition['sampled_qp_execution_alpha'][order,0],expected[order])


def test_blended_acceleration_force_offset_gradients_and_failed_isolation():
    torch.manual_seed(3)
    mass=torch.eye(18).expand(3,-1,-1).clone(); mass[:,0,6]=mass[:,6,0]=.2
    nominal=torch.randn(3,12,requires_grad=True)
    candidate=nominal*2
    external=torch.ones(3,18)*.3
    qdd=torch.linalg.solve(mass,(torch.cat((torch.zeros(3,6),candidate),-1)+external)[...,None]).squeeze(-1)
    alpha=torch.tensor([[0.],[.4],[1.]])
    selector=torch.cat((torch.zeros(3,6,12),torch.eye(12).expand(3,-1,-1)),1)
    torque_map=torch.linalg.solve(mass,selector)
    actual=blended_candidate_acceleration(qdd,candidate,nominal,alpha,torque_map,torch.tensor([1,1,0]))
    executed=nominal+alpha*(candidate-nominal)
    expected=torch.linalg.solve(mass,(torch.cat((torch.zeros(3,6),executed),-1)+external)[...,None]).squeeze(-1)
    torch.testing.assert_close(actual[:2],expected[:2])
    assert not torch.allclose(actual[:2],alpha[:2]*qdd[:2])
    actual.square().sum().backward()
    assert nominal.grad.isfinite().all() and nominal.grad[:2].abs().sum()>0 and nominal.grad[2].eq(0).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='cuPIQP CUDA unavailable')
def test_recovery_blended_qdd_preserves_full_candidate_and_masks_failed_rows():
    from test_hard_pact_reduced_qp import inputs, solver
    from rsl_rl.algorithms.hard_pact_qp import recovery_projection_loss
    d=inputs(3,device='cuda'); d['joint_position'][1,0]=2.4
    d['joint_position'][2,0]=float('nan'); d['tau_nom'].requires_grad_()
    qp=solver(qp_solver='cupiqp',solver_dtype='float64',rollout_max_iter=100,ppo_max_iter=100)
    reference=qp.solve(differentiable=True,**d)
    alpha=d['tau_nom'].new_tensor([[0.],[.4],[1.]])
    out=qp.solve(differentiable=True,execution_base=d['tau_nom'],execution_alpha=alpha,**d)
    assert out.stage.tolist()==[0,1,2]
    torch.testing.assert_close(out.tau_safe,reference.tau_safe,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(out.qdd,reference.qdd,atol=1e-5,rtol=1e-5)
    selected=execution_torque(d['tau_nom'],out.tau_safe,out.differentiated_mask|out.recovery_mask,alpha)
    # Identity inertia, zero contact Jacobians/bias in this fixture.
    torch.testing.assert_close(out.execution_qdd[:2,6:],selected[:2])
    assert out.execution_qdd[2].eq(0).all()
    out.execution_qdd.square().sum().backward()
    assert d['tau_nom'].grad.isfinite().all() and d['tau_nom'].grad[2].eq(0).all()


def test_checkpoint_strength_resume_and_old_disabled_contract(tmp_path):
    from test_training_state_transfer import hard
    from rsl_rl.utils import hard_pact_checkpoint as hc
    from rsl_rl.utils import training_checkpoint as tc
    from rsl_rl.algorithms.hard_pact_qp_curriculum import QPCurriculum
    from legged_gym.envs.go2.go2_hard_pact.deployment import qp_update_contract
    from dataclasses import asdict
    cfg=HardPACTQPConfig(qp_update_mode='curriculum_substeps')
    def configured(settings):
        r=hard(); r.alg.qp_config=settings; r.qp_curriculum=QPCurriculum(settings)
        r.train_cfg['algorithm']['hard_pact_qp']=asdict(settings)
        r.deployment_contract['qp_update']=qp_update_contract('curriculum_substeps',4,qp_config=settings)
        return r
    source=configured(cfg)
    source.qp_curriculum.intervention.progress=.4
    path=tmp_path/'old.pt'; source.save(path)
    saved=torch.load(path,weights_only=False)
    def remove_new_fields(value):
        if isinstance(value,dict):
            value.pop('intervention_extra_strength_enabled',None)
            value.pop('intervention_extra_strength_start',None)
            for child in value.values(): remove_new_fields(child)
    remove_new_fields(saved[tc.KEY]['contract']); torch.save(saved,path)
    target=configured(cfg)
    hc.load(target,path)
    assert target.qp_curriculum.intervention.progress==.4
    assert not target.alg.qp_config.intervention_extra_strength_enabled
    incompatible=configured(replace(cfg,intervention_extra_strength_enabled=True))
    with pytest.raises(ValueError): hc.load(incompatible,path)
    source=configured(replace(cfg,intervention_extra_strength_enabled=True))
    source.qp_curriculum.intervention.progress=.6
    enabled=tmp_path/'enabled.pt'; source.save(enabled)
    target=configured(source.alg.qp_config); hc.load(target,enabled)
    assert target.qp_curriculum.intervention.progress==.6
    assert target.alg.qp_config.intervention_extra_strength_enabled


@pytest.mark.skipif(not torch.cuda.is_available(),reason='cuPIQP CUDA unavailable')
def test_tiny_ppo_enabled_velocity_replay(monkeypatch):
    import test_hard_pact_contact_indexing_and_inverse_gate as live
    from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
    update=PPO_HardPACT.update
    def wrapped(self,*args,**kwargs):
        self.qp_config=replace(self.qp_config,intervention_extra_strength_enabled=True)
        self.hard_pact_qp.cfg=self.qp_config
        f=self.storage.hard_pact_fields
        f['sampled_qp_execution_alpha']=torch.tensor([0.,.25,.6,1.],device=self.device).reshape(1,4,1).expand(2,-1,-1).clone()
        f['sampled_qp_base_rate_clip']=torch.zeros(2,4,1,device=self.device,dtype=torch.bool)
        self.lambda_qp_velocity_xy=self.lambda_qp_velocity_yaw=.1
        return update(self,*args,**kwargs)
    monkeypatch.setattr(PPO_HardPACT,'update',wrapped)
    with torch.device('cuda'): live._run_owned_update('curriculum_substeps','cuda')
