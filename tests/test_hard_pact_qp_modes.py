"""Control-loop and current-parameter replay checks using the real 24-D QP."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact.deployment import qp_update_contract, validate_qp_deployment_contract
from rsl_rl.algorithms.hard_pact_qp import project_nominal_torque, HardPACTQPConfig, balanced_substep_indices
from rsl_rl.algorithms.ppo_hard_pact import replay_qp_torques, disjoint_qp_epoch_mask
from test_hard_pact_reduced_qp import inputs, solver


class Heads(torch.nn.Module):
    def __init__(self):
        super().__init__();self.gain=torch.nn.Parameter(torch.tensor(.01));self.calls=0
    def predict_grf(self,z,e,tau):
        self.calls+=1
        f=tau*self.gain
        return f+tau.new_tensor([0.,0.,.08]*4)
    def grf_to_physical(self,x):return x*250
    def wrench_to_qp_physical(self,x):return x


def fixture(mode,n=8):
    task=Go2HardPACT.__new__(Go2HardPACT)
    task.num_envs=n;task.device=torch.device("cpu")
    task.cfg=SimpleNamespace(sim=SimpleNamespace(dt=.01),control=SimpleNamespace(decimation=4))
    q=torch.zeros(n,12);v=q.clone();quat=torch.tensor([0.,0.,0.,1.]).expand(n,-1).clone()
    task._canonical_joint_state=lambda:(q,v)
    task._canonical_configuration=lambda *_:torch.cat((torch.zeros(n,3),quat,q),1)
    task._canonical_velocity_world=lambda:torch.cat((torch.zeros(n,6),v),1)
    task.simulator=SimpleNamespace(_torques=q.clone())
    task.simulator.hard_pact_set_executed_torque=lambda tau:setattr(task.simulator,"_torques",tau.clone())
    task._legacy_task_class=SimpleNamespace(reset_idx=lambda *_:None)
    task._hard_pact_previous_substep_torque=q.clone()
    task._hard_pact_q_d=torch.full_like(q,.05);task._hard_pact_tau_ff=q.clone()
    task._get_pinn_feedback=lambda d,p,v:3*(d-p)-v
    task._hard_pact_policy_latent=torch.zeros(n,16)
    task._hard_pact_policy_explicit=torch.zeros(n,11);task._hard_pact_policy_explicit[:,3:7]=.8
    task._hard_pact_policy_explicit[:,3]=.2
    task._hard_pact_wrench_raw_normalized=torch.zeros(n,6)
    heads=Heads();task._hard_pact_actor_critic=SimpleNamespace(physics_estimator=heads)
    qp=solver(qp_update_mode=mode,torque_rate_limit_nm_s=10.)
    task._hard_pact_rollout_qp=qp
    counts=torch.zeros(n,dtype=torch.long);dispatch=[];build_rows=[]
    real=qp.solve
    def solve(**kw):
        ids=kw["environment_ids"];counts[ids]+=1;dispatch.append(ids.clone())
        assert not torch.is_grad_enabled()
        return real(**kw)
    qp.solve=solve
    def mechanics(state,velocity,**kw):
        build_rows.append(state.shape[0]);d=inputs(state.shape[0],state.dtype)
        return SimpleNamespace(**{k:d[k] for k in ("mass_matrix","bias","foot_jacobians","base_jacobian","foot_acceleration_bias")})
    task._hard_pact_bard_dynamics=SimpleNamespace(build_context=mechanics)
    return task,heads,qp,real,counts,q,v,quat,build_rows


@pytest.mark.parametrize("mode,per_interval",[("every_substep",4),("random_one_substep",1)])
def test_control_reset_bounds_no_held_correction_and_current_parameter_replay(mode,per_interval):
    task,heads,qp,real,counts,q,v,quat,build_rows=fixture(mode)
    task._begin_qp_interval();task.reset_idx(torch.arange(8));task._hard_pact_q_d.fill_(.05)
    for interval in range(3):
        q.zero_();v.zero_();quat[:,2]=0.;quat[:,3]=1.
        before=heads.calls
        task._begin_qp_interval()
        bins=torch.bincount(task._qp_sampled_substep_index.long(),minlength=4)
        assert bins.tolist()==[2]*4
        for k in range(4):
            q.fill_(.005*k);v.fill_(.001*k)
            quat[:,2]=torch.sin(torch.tensor(.1*k));quat[:,3]=torch.cos(torch.tensor(.1*k))
            prev=task._hard_pact_previous_substep_torque.clone()
            nominal=task._get_pinn_feedback(task._hard_pact_q_d,q,v)
            task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(8,6))
            assert heads.calls==before+1  # NO neural forward in any substep
            tau=task.simulator._torques
            assert torch.isfinite(tau).all() and not tau.requires_grad
            assert ((tau-prev).abs()<=.100001).all()
            assert (tau.abs()<=qp.torque_limits).all()
            unsolved=task._qp_sampled_substep_index!=k
            if mode=="random_one_substep":
                expected=project_nominal_torque(nominal,prev,qp.torque_limits,10.,.01)
                torch.testing.assert_close(tau[unsolved],expected[unsolved],rtol=0,atol=0)
        assert heads.calls-before==1
        assert counts.tolist()==[per_interval*(interval+1)]*8
        assert task._qp_interval_solve_count.flatten().tolist()==[per_interval]*8
        packet=task._qp_sampled_transition
        assert packet["sampled_qp_valid"].all()
        desired=task._hard_pact_q_d.clone().requires_grad_();ff=task._hard_pact_tau_ff.clone().requires_grad_()
        nominal,condition=replay_qp_torques(desired,ff,packet,task._get_pinn_feedback)
        torch.testing.assert_close(nominal,packet["sampled_qp_rollout_nominal_torque"],rtol=0,atol=0)
        torch.testing.assert_close(condition,packet["sampled_qp_grf_conditioning_torque"],rtol=0,atol=0)
        force=task._yaw_local_to_world(heads.grf_to_physical(
            heads.predict_grf(None,None,condition)).reshape(8,4,3),packet["sampled_qp_q"][:,3:7])
        torch.testing.assert_close(force.flatten(1),packet["sampled_qp_rollout_grf_world"],rtol=0,atol=0)
        data=inputs(8,torch.float32)
        data.update(tau_nom=nominal,force_pred_world=force,contact_probability=packet["sampled_qp_rollout_contact_probability"],
            base_quaternion=packet["sampled_qp_q"][:,3:7],
            previous_torque=packet["sampled_qp_previous_torque"],joint_position=packet["sampled_qp_q"][:,7:],
            joint_velocity=packet["sampled_qp_v"][:,6:])
        out=real(differentiable=True,**data)
        torch.testing.assert_close(out.tau_safe,packet["sampled_qp_safe_torque"],rtol=1e-6,atol=1e-7)
        (out.tau_safe.sum()+out.force_world.sum()*.01).backward()
        assert desired.grad.abs().sum()>0 and torch.isfinite(desired.grad).all()
        assert heads.gain.grad.abs()>0
    assert sum(build_rows)==counts.sum()
    task.reset_idx(torch.tensor([1,3]));assert task._hard_pact_previous_substep_torque[[1,3]].eq(0).all()
    assert not task._qp_sampled_transition["sampled_qp_valid"][[1,3]].any()


def test_contract_rejects_retired_modes_and_partition_preserved():
    assert HardPACTQPConfig().qp_update_mode=="random_one_substep"
    for mode in ("every_substep","random_one_substep"):
        c={"schema_version":14,"qp_update":qp_update_contract(mode,4)}
        assert validate_qp_deployment_contract(c) is c
        assert c["qp_update"]["ppo_projection_loss_multiplier"]==1
    for mode in ("active_constraint_update","two_anchor_held_correction","single_anchor_held_correction"):
        with pytest.raises(ValueError):solver(qp_update_mode=mode)
    with pytest.raises(ValueError):validate_qp_deployment_contract({"schema_version":11})
    indices=torch.arange(100);anchors=balanced_substep_indices(100,4,"cpu",generator=torch.Generator().manual_seed(7))
    masks=[disjoint_qp_epoch_mask(indices,anchors,epoch=e,num_epochs=5) for e in range(5)]
    assert torch.stack(masks).sum(0).eq(1).all()
