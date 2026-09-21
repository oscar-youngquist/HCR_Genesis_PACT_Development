"""Throttling, reset-safe measured derivatives and count denominators."""
from types import SimpleNamespace
import torch
import pytest
from rsl_rl.algorithms.hard_pact_qp_diagnostics import QPIterationDiagnostics, QPMeasuredMotion
from rsl_rl.runners.pact_runner import OnPolicyRunnerPACT
from test_hard_pact_reduced_qp import solver
from test_hard_pact_reduced_qp import inputs, objective

def test_incompatible_formulation_configuration_rejected():
    from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig
    with pytest.raises(ValueError,match="Incompatible HardPACT QP metadata"):
        HardPACTQPConfig.from_dict({"slack_weight":200.})

def test_physical_objective_components_match_direct_formula():
    qp=solver(diagnostics_level="physical");data=inputs(2)
    data['foot_jacobians'][:,:,:,:3]=torch.eye(3,dtype=torch.float64)
    data['base_angular_velocity_world'].fill_(.2)
    problem=qp._build(data);x=torch.randn(2,24,dtype=torch.float64)
    metrics=qp._physical_diagnostics(problem,x,data)
    total=sum(value for key,value in metrics.items() if key.startswith('objective/'))
    ridge=.5*qp.cfg.q_regularization*(x/problem.variable_scale).square().sum(-1)
    torch.testing.assert_close(total+ridge,objective(qp,data,problem,x),rtol=1e-12,atol=1e-12)

def test_measured_derivatives_use_physics_dt_and_reset_stencils():
    motion=QPMeasuredMotion();agg=QPIterationDiagnostics()
    q=torch.zeros(2,12);v=q.clone();tau=q.clone()
    def sample():
        motion.update(agg,q,v,tau,.01,-torch.ones(12),torch.ones(12),torch.ones(12)*3,torch.tensor([True,False]))
    sample();v+=.02;tau+=.03;sample();v+=.04;tau+=.03;sample()
    metrics=agg.finalize(q)
    torch.testing.assert_close(metrics['measured/all/acceleration_abs_rad_s2'],torch.tensor(3.))
    torch.testing.assert_close(metrics['measured/all/jerk_abs_rad_s3'],torch.tensor(200.))
    motion.reset(torch.arange(2));agg=QPIterationDiagnostics();v.fill_(999);sample()
    assert torch.isnan(agg.finalize(q)['measured/all/acceleration_abs_rad_s2'])
    assert torch.isnan(agg.finalize(q)['measured/all/jerk_abs_rad_s3'])

def test_counts_use_environment_intervals_not_dispatches():
    agg=QPIterationDiagnostics();ref=torch.zeros(1)
    agg.add_sum('environment_control_intervals',torch.tensor(8))
    agg.add_sum('real_rows',torch.tensor(8));agg.add_sum('solve_calls',torch.tensor(4))
    for k in range(4):agg.add_sum(f'sampled_substep/{k}_count',torch.tensor(2))
    metrics=agg.finalize(ref)
    assert metrics['problems_per_environment_control_interval']==1
    assert metrics['solved_substep_coverage']==.25
    assert metrics['sampled_substep/0_fraction']==.25

def test_runner_throttle_has_no_writes_when_unscheduled():
    qp=solver(tensorboard_diagnostics_interval=3)
    runner=OnPolicyRunnerPACT.__new__(OnPolicyRunnerPACT)
    runner.alg=SimpleNamespace(hard_pact_qp=qp,qp_enabled_at_iteration=lambda:True,last_qp_metrics={})
    runner.env=SimpleNamespace(set_hard_pact_qp_enabled=lambda _:None)
    calls=[];runner.writer=SimpleNamespace(add_scalar=lambda *x:calls.append(x))
    runner._set_hard_pact_qp_iteration(1);runner._log_qp_metrics(1)
    assert not calls and not qp.diagnostics_scheduled
    runner._set_hard_pact_qp_iteration(3);runner._log_qp_metrics(3)
    assert calls and qp.diagnostics_scheduled
    assert len(calls)==len({x[0] for x in calls})
    for phase in ('rollout','ppo'):
        assert f'qp/{phase}/attempt/soft_joint_fraction' in {x[0] for x in calls}

def test_soft_joint_recovery_rates_aggregate_counts_before_division():
    agg=QPIterationDiagnostics();ref=torch.zeros(1)
    # Unequal chunks: 2/2 and 1/8 attempts, two successes, one exception.
    for rows,attempts,successes,exceptions in ((2,2,1,1),(8,1,1,0)):
        for key,value in (('real_rows',rows),('attempt/soft_joint_count',attempts),
                          ('final/soft_joint_count',successes),
                          ('attempt/soft_joint_exception_count',exceptions)):
            agg.add_sum(key,torch.tensor(value))
    metrics=agg.finalize(ref)
    for key,value in (('attempt/soft_joint_fraction',.3),
                      ('attempt/soft_joint_success_fraction',2/3),
                      ('attempt/soft_joint_failure_fraction',1/3),
                      ('attempt/soft_joint_exception_fraction',1/3),
                      ('final/soft_joint_fraction',.2)):
        torch.testing.assert_close(metrics[key],torch.tensor(value))
    empty=QPIterationDiagnostics().finalize(ref)
    assert all(empty[k]==0 for k in metrics if k.startswith('attempt/soft_joint') and k.endswith('fraction'))
