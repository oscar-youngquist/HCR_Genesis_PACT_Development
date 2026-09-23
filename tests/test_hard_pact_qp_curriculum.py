"""Independent schedules and real control/QP paths; no simulator or training."""
from dataclasses import replace
from types import SimpleNamespace
import math

import pytest
import torch

from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig, projection_loss
from rsl_rl.algorithms.hard_pact_qp_curriculum import QPCurriculum, execution_torque
from test_hard_pact_qp_modes import fixture
from test_hard_pact_reduced_qp import inputs, solver


def config(**kw):
    return replace(HardPACTQPConfig(), correction_ramp_enabled=True,
                   objective_curriculum_enabled=True, **kw)


def test_ramp_endpoints_and_disabled():
    s=QPCurriculum(config(warmup_iterations=20,correction_ramp_duration=10))
    assert [s.begin(i)[1] for i in (0,20,25,30,40)] == [0,0,.5,1,1]
    assert s.start == 30
    assert QPCurriculum(config(correction_ramp_duration=0)).begin(0)[1] == 1
    c=HardPACTQPConfig()
    assert QPCurriculum(c).begin(0) == (c,1.)


def test_performance_missing_once_snapshot_and_resume():
    c=config(correction_ramp_duration=2,objective_curriculum_step_interval=2,
             objective_curriculum_ema_alpha=1.,objective_curriculum_progress_delta=.2,
             objective_curriculum_min_samples=4)
    s=QPCurriculum(c)
    for i,p,n in [(0,.9,4),(1,.9,4),(2,None,4),(3,float('nan'),4),(4,.9,3)]:
        s.begin(i);s.finish(i,p,n)
        assert s.progress == 0
    snapshot=s.begin(5)
    s.finish(5,.9,4)
    assert s.progress == .2 and s.begin(5) is snapshot
    s.finish(5,.9,4)
    assert s.progress == .2
    restored=QPCurriculum(c);restored.load_state_dict(s.state_dict())
    for i,p in [(6,.9),(7,.1),(8,.9),(9,.9)]:
        assert s.begin(i)==restored.begin(i)
        s.finish(i,p,4);restored.finish(i,p,4)
        assert s.state_dict()==restored.state_dict()
    assert math.isclose(s.progress,.4)


def test_execution_endpoints_fallback_and_full_loss_gradients():
    candidate=torch.tensor([[4.]*12,[8.]*12,[2.]*12],requires_grad=True)
    nominal=torch.ones_like(candidate,requires_grad=True)
    accepted=torch.tensor([True,True,False])
    gradients=[]
    for alpha in (0.,.5,1.):
        actual=execution_torque(nominal,candidate,accepted,alpha)
        assert torch.equal(actual[2],candidate[2])  # deterministic fallback untouched
        torch.testing.assert_close(actual[:2],nominal[:2]+alpha*(candidate[:2]-nominal[:2]))
        loss=projection_loss(candidate,nominal,torch.ones(12),torch.ones(3,dtype=torch.bool),accepted)
        gradients.append(torch.autograd.grad(loss,(candidate,nominal),retain_graph=True))
    for gradient in gradients[1:]:
        for a,b in zip(gradient,gradients[0]):torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('mode',['every_substep','random_one_substep'])
@pytest.mark.parametrize('alpha',[0.,.5,1.])
def test_control_replay_actual_torque_and_counts(mode,alpha):
    task,heads,qp,real,counts,q,v,quat,_=fixture(mode)
    task._qp_execution_alpha=alpha
    task._begin_qp_interval()
    task.reset_idx(torch.arange(task.num_envs))
    for interval in range(2):
        task._hard_pact_q_d.fill_(.05)
        task._begin_qp_interval()
        for k in range(4):
            task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(8,6))
            assert torch.isfinite(task.simulator._torques).all()
            # The callback's executed history is exactly the blended command.
            torch.testing.assert_close(task._hard_pact_previous_substep_torque,task.simulator._torques)
        packet=task._qp_sampled_transition
        from rsl_rl.algorithms.hard_pact_qp import project_nominal_torque
        base=project_nominal_torque(packet['sampled_qp_rollout_nominal_torque'],
            packet['sampled_qp_previous_torque'],qp.torque_limits,10.,.01)
        expected=execution_torque(base,packet['sampled_qp_safe_torque'],
                                 packet['sampled_qp_differentiated'].flatten().bool(),alpha)
        torch.testing.assert_close(expected,packet['sampled_qp_executed_torque'])
    assert counts.tolist()==[8 if mode=='every_substep' else 2]*8


def test_weights_update_assembly_without_invalidating_backend(monkeypatch):
    qp=solver()
    d=inputs(2)
    d['foot_jacobians'][:,:,0,6] = .1
    d['contact_probability'].fill_(1.)
    before=qp._build(d)
    instances=qp._backend_instances
    monkeypatch.setattr(qp,'clear_warm_start',lambda:pytest.fail('weights invalidated solver caches'))
    qp.cfg=replace(qp.cfg,contact_acceleration_weight=.25,attitude_weight=.25)
    after=qp._build(d)
    assert not torch.equal(before.Q,after.Q)
    result=qp.solve(differentiable=False,**d)
    assert qp._backend_instances is instances
    assert torch.isfinite(result.tau_safe).all()


def test_tracking_collection_independent_of_other_curricula():
    from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
    task=Go2HardPACT.__new__(Go2HardPACT)
    task.cfg=SimpleNamespace(commands=SimpleNamespace(curriculum=False,curriculum_patience_iterations=0))
    task.device='cpu';task._qp_tracking_collect=True
    task._legacy_task_class=SimpleNamespace(_reward_tracking_lin_vel=lambda _:torch.tensor([.2,.8]))
    task.begin_command_curriculum_iteration()
    task._reward_tracking_lin_vel();task._reward_tracking_lin_vel()
    assert task._command_tracking_count == 4
    assert task._command_tracking_sum.item() == 2.


def test_shared_raw_performance_ignores_episode_lengths_and_command_bounds():
    from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
    from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
    task=Go2HardPACT.__new__(Go2HardPACT)
    task.device='cpu';task._qp_tracking_collect=False
    task.cfg=SimpleNamespace(commands=SimpleNamespace(curriculum=False,curriculum_patience_iterations=0))
    task.simulator=SimpleNamespace(use_domainrand_curriculum=True,base_lin_vel=torch.zeros(2,3))
    task.commands=torch.tensor([[.2,0.,0.],[.4,0.,0.]])
    task._legacy_task_class=Go2PACT
    results=[]
    for episode_length,bounds,scale in [(2.,[-1.,1.],1.),(100.,[-8.,8.],20.)]:
        task.max_episode_length_s=episode_length
        task.command_ranges={'lin_vel_x':bounds};task.reward_scales={'tracking_lin_vel':scale}
        task.begin_command_curriculum_iteration()
        raw=task._reward_tracking_lin_vel()
        results.append(task.locomotion_curriculum_performance())
        assert results[-1] == (raw.mean().item(),2)
    assert results[0] == results[1]
    task._command_tracking_sum.fill_(float('nan'))
    assert task.locomotion_curriculum_performance() == (None,2)
    task.simulator.use_domainrand_curriculum=False
    task.begin_command_curriculum_iteration()
    assert task.locomotion_curriculum_performance() == (None,0)


def test_runner_snapshot_shared_rollout_and_ppo():
    from rsl_rl.runners.pact_runner import OnPolicyRunnerPACT
    qp=solver()
    qp.cfg=config(correction_ramp_duration=0,objective_curriculum_step_interval=1)
    runner=SimpleNamespace(alg=SimpleNamespace(hard_pact_qp=qp,qp_enabled_at_iteration=lambda:True),
        env=SimpleNamespace(set_hard_pact_qp_enabled=lambda _:None))
    OnPolicyRunnerPACT._set_hard_pact_qp_iteration(runner,0)
    assert qp.cfg.contact_acceleration_weight == .25
    frozen=qp.cfg
    runner.qp_curriculum.finish(0,.9,10)
    OnPolicyRunnerPACT._set_hard_pact_qp_iteration(runner,0)
    assert qp.cfg is frozen  # even repeated setup within an iteration is immutable
    OnPolicyRunnerPACT._set_hard_pact_qp_iteration(runner,1)
    assert qp.cfg.contact_acceleration_weight > .25
    assert runner.env._qp_execution_alpha == 1.
