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
