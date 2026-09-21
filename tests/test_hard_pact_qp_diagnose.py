"""Bounded capture without simulator; CPU solves only test plumbing, not cuPIQP."""
from unittest.mock import patch
import torch
import pytest
from rsl_rl.algorithms.hard_pact_qp_backends import QPBackendResult
from rsl_rl.algorithms.hard_pact_qp_diagnose import QPCapture, independent_checks
from test_hard_pact_reduced_qp import inputs, solver
from scripts.diagnose_hard_pact_qp import replay_one


def owner():
    qp = solver(soft_joint_recovery_enabled=False)
    qp._active_solver = "qpth"
    qp._active_differentiable = True
    qp._diagnostics_phase = "ppo"
    return qp


@pytest.mark.parametrize("stage", ["primary", "recovery"])
def test_roundtrip_extreme_finite_owned_inputs_stage_and_budget(tmp_path, stage):
    qp, d = owner(), inputs(2)
    m = qp._build(d)
    if stage == "recovery":
        m = qp._soft_joint_problem(m)
    capture = QPCapture(tmp_path, limit=1, healthy_limit=0)
    packet = capture.before(qp,m,d,stage,torch.tensor([4,8]))
    original = m.Q.clone()
    m.Q.add_(1)  # emulate solver mutation AFTER snapshot
    z = torch.zeros_like(m.p)
    z[1,0] = 3.6e16
    capture.after(packet,QPBackendResult(z),torch.tensor([True,False]))
    path, = tmp_path.glob("*.pt")
    saved = torch.load(path,weights_only=True)
    assert saved["stage"] == stage and saved["phase"] == "ppo"
    assert saved["rows"].tolist() == [4,8]
    assert saved["failing_rows"].tolist() == [1]
    torch.testing.assert_close(saved["tensors"]["Q"],original)
    assert saved["raw_torque_violation_nm"][1] > 1e16
    assert capture.before(qp,m,d,stage,torch.arange(2)) is None
    assert capture.bytes_written == path.stat().st_size


@pytest.mark.parametrize("trigger", ["nan", "status", "exception"])
def test_triggers(tmp_path, trigger):
    qp,d=owner(),inputs(2)
    capture=QPCapture(tmp_path,healthy_limit=0)
    packet=capture.before(qp,qp._build(d),d,"primary",torch.arange(2))
    z=torch.zeros(2,24,dtype=torch.float64)
    if trigger=="nan":z[0,0]=float("nan")
    if trigger=="exception":capture.after(packet,error=RuntimeError("numerical"))
    else:capture.after(packet,QPBackendResult(z,snapshot={"status":torch.tensor([4,0])} if trigger=="status" else None),torch.zeros(2,dtype=torch.bool))
    assert capture.count==1


def test_disabled_and_enabled_capture_forward_gradient_parity(tmp_path):
    results=[]
    for enabled in (False,True):
        qp,d=owner(),inputs(2)
        d["tau_nom"].requires_grad_()
        if enabled:qp.diagnostic_capture=QPCapture(tmp_path)
        out=qp.solve(differentiable=True,**d)
        out.tau_safe.sum().backward()
        results.append((out.tau_safe.detach(),d["tau_nom"].grad))
    for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=0,atol=0)
    path, = tmp_path.glob("*.pt")
    packet=torch.load(path,weights_only=True)
    assert packet["healthy_reference"]
    report=replay_one(packet,"cpu","baseline")
    assert report["certified_rows"]==2 and report["certified_finite"]
    assert report["mixed_finite"] and report["mixed_excluded_zero"]


def test_independent_checks_and_byte_limit(tmp_path):
    qp,d=owner(),inputs(2)
    capture=QPCapture(tmp_path,byte_limit=1)
    assert capture.before(qp,qp._build(d),d,"primary",torch.arange(2)) is None
    assert capture.dropped==1 and not list(tmp_path.iterdir())
    capture=QPCapture(tmp_path)
    packet=capture.before(qp,qp._build(d),d,"primary",torch.arange(2))
    z=torch.zeros(2,24,dtype=torch.float64)
    z[1,0]=1e6
    result=independent_checks(packet,z)
    assert result["certified"].tolist()==[True,False]


def test_capture_integration_primary_recovery_and_history(tmp_path):
    qp,d=owner(),inputs(1)
    from dataclasses import replace
    qp.cfg=replace(qp.cfg,soft_joint_recovery_enabled=True)
    qp.diagnostic_capture=QPCapture(tmp_path,healthy_limit=0,history_limit=1)
    def failure(m):
        z=torch.zeros_like(m.p)
        z[:,0]=1e16
        z[:,14]=-1  # negative normal force remains infeasible after torque clamp
        return QPBackendResult(z)
    with patch.object(qp,"_backend_solve",side_effect=failure):
        qp.solve(differentiable=False,**d)
    files=sorted(tmp_path.glob("*.pt"))
    packets=[torch.load(p,weights_only=True) for p in files]
    assert [p["stage"] for p in packets]==["primary","recovery"]
    assert packets[1]["preceding_updates"][0]["stage"]=="primary"
    assert packets[0]["raw_primal"].shape[-1]==24
    assert packets[1]["raw_primal"].shape[-1]==36


def test_backend_capture_owned_status_iterations_and_disabled_fast_exit():
    import numpy as np
    from types import SimpleNamespace
    from rsl_rl.algorithms.hard_pact_qp_backends import _capture_solver_details
    backend=SimpleNamespace(capture_details_enabled=False)
    _capture_solver_details(backend,None,None,2,False)  # no access when disabled
    backend.capture_details_enabled=True
    source=torch.ones(2,24)
    info=SimpleNamespace(status_value=np.array([4,0]),iter=np.array([30,2]))
    solver=SimpleNamespace(result=SimpleNamespace(x=source,info=info))
    _capture_solver_details(backend,solver,source,2,True)
    source.zero_()
    assert backend.last_capture_details["status"].tolist()==[4,0]
    assert backend.last_capture_details["iter"].tolist()==[30,2]
    assert backend.last_capture_details["x"].eq(1).all()
