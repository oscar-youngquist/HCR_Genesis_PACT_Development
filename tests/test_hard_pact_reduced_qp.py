"""24-D torque/force formulation: physical algebra, certificates and VJPs."""
from dataclasses import replace
from unittest import mock

import pytest
import torch

from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig, HardPACTDifferentiableQP, projection_loss
from rsl_rl.algorithms.hard_pact_qp_backends import QPBackendResult


def inputs(batch=2, dtype=torch.float64, device="cpu"):
    z = lambda *s: torch.zeros(s, dtype=dtype, device=device)
    d = dict(mass_matrix=torch.eye(18,dtype=dtype,device=device).expand(batch,-1,-1).clone(),
             bias=z(batch,18), foot_jacobians=z(batch,4,3,18), base_jacobian=z(batch,6,18),
             foot_acceleration_bias=z(batch,4,3), tau_nom=z(batch,12),
             force_pred_world=z(batch,4,3), wrench_pred_world=z(batch,6),
             contact_probability=torch.ones(batch,4,dtype=dtype,device=device),
             previous_torque=z(batch,12),joint_position=z(batch,12),joint_velocity=z(batch,12),
             dt=torch.full((batch,1),.01,dtype=dtype,device=device),
             base_quaternion=z(batch,4),base_angular_velocity_world=z(batch,3))
    d["base_quaternion"][:,3]=1
    d["base_jacobian"][:,:,:6]=torch.eye(6,dtype=dtype,device=device)
    d["force_pred_world"][:,:,2]=10
    d["tau_nom"].fill_(.4)
    return d


def solver(**kw):
    cfg=HardPACTQPConfig(**dict(dict(max_iter=60,not_improved_limit=10,
        exception_capture_enabled=False),**kw))
    return HardPACTDifferentiableQP(cfg,[23.5]*12,[-2]*12,[2]*12,[30]*12)


def test_outer_projection_stance_mask_invalid_rows_and_empty_batch():
    tau=torch.ones(3,12,dtype=torch.float64,requires_grad=True)
    nominal=torch.zeros_like(tau,requires_grad=True)
    acc=torch.ones(3,18,dtype=torch.float64,requires_grad=True)
    jac=torch.zeros(3,4,3,18,dtype=torch.float64)
    jac[:,:,:,:3]=torch.eye(3,dtype=torch.float64)
    jac.requires_grad_()
    bias=torch.zeros(3,4,3,dtype=torch.float64,requires_grad=True)
    stance=torch.tensor([[1.,0.,0.,0.]]*3,requires_grad=True)
    with torch.no_grad():
        tau[1:]=float('nan');acc[1:]=float('nan');jac[1:]=float('nan')
    kwargs=dict(qdd=acc,foot_jacobians=jac,foot_acceleration_bias=bias,
                stance_mask=stance,contact_weight=2.,contact_scale=2.)
    loss=projection_loss(tau,nominal,torch.ones(12),torch.tensor([1,1,0]),
                         torch.tensor([1,0,1]),**kwargs)
    torch.testing.assert_close(loss,torch.tensor(13.5,dtype=torch.float64))
    loss.backward()
    for value in (tau,nominal,acc):
        assert value.grad.isfinite().all() and value.grad[1:].eq(0).all()
    assert acc.grad[0,:3].abs().sum()>0
    assert jac.grad is None and bias.grad is None and stance.grad is None
    empty=projection_loss(tau,nominal,torch.ones(12),torch.zeros(3),torch.ones(3),**kwargs)
    assert empty.isfinite() and empty==0
    for value in (tau,nominal,acc):value.grad=None
    empty.backward()
    assert all(value.grad.eq(0).all() for value in (tau,nominal,acc))


def test_outer_projection_wrench_clamp_gradient():
    from rsl_rl.modules.hard_pact_physics import sanitize_and_clip_wrench_for_qp
    qp=solver(); d=inputs(1)
    # Couple base acceleration into contact acceleration so the outer loss
    # sees the wrench, while the ordinary physical clamp blocks saturated axes.
    d['foot_jacobians'][:,:,:3,:3]=torch.eye(3,dtype=torch.float64)
    raw=torch.tensor([[200.,1.,1.,0.,0.,0.]],dtype=torch.float64,requires_grad=True)
    d['wrench_pred_world']=sanitize_and_clip_wrench_for_qp(raw,torch.tensor([150.,150.,150.,40.,40.,40.]))
    out=qp.solve(differentiable=True,**d)
    assert out.differentiated_mask.all()
    loss=projection_loss(out.tau_safe,d['tau_nom'],qp.torque_limits,torch.ones(1),out.differentiated_mask,
        qdd=out.qdd,foot_jacobians=d['foot_jacobians'],foot_acceleration_bias=d['foot_acceleration_bias'],
        stance_mask=d['contact_probability']>=.5,contact_weight=1.,contact_scale=50.)
    loss.backward()
    assert raw.grad.isfinite().all() and raw.grad[0,0]==0 and raw.grad[0,1:3].abs().sum()>0


def objective(qp,d,m,x):
    a=(m.acceleration_map@x[...,None]).squeeze(-1)+m.acceleration_offset
    mask=d["contact_probability"]>=qp.cfg.contact_threshold
    f=torch.where(mask[...,None],d["force_pred_world"],0).flatten(1)
    total=qp.cfg.torque_tracking_weight*((x[:,:12]-d["tau_nom"])/qp.cfg.torque_scale_nm).square().sum(-1)
    total+=qp.cfg.force_tracking_weight*((x[:,12:]-f)/qp.cfg.force_scale_n).square().sum(-1)
    ca=(d["foot_jacobians"]@a[:,None,:,None]).squeeze(-1)+d["foot_acceleration_bias"]
    total+=qp.cfg.contact_acceleration_weight*(ca/qp.cfg.contact_acceleration_scale_m_s2).square().sum(-1).mul(mask).sum(-1)
    # Identity attitude in this independent algebra fixture.
    desired=-qp.cfg.attitude_kd*d["base_angular_velocity_world"][:,:2]
    aa=(d["base_jacobian"][:,3:5]@a[...,None]).squeeze(-1)
    total+=qp.cfg.attitude_weight*((aa-desired)/qp.cfg.attitude_acceleration_scale_rad_s2).square().sum(-1)
    total+=.5*qp.cfg.q_regularization*(x/m.variable_scale).square().sum(-1)
    return total


def test_affine_acceleration_and_objective_spd():
    torch.manual_seed(7)
    d=inputs(); t=torch.randn(2,18,18,dtype=torch.float64)
    d["mass_matrix"]=t@t.transpose(1,2)+torch.eye(18)*2
    d["foot_jacobians"].normal_(std=.1); d["bias"].normal_()
    d["wrench_pred_world"].normal_(); d["foot_acceleration_bias"].normal_()
    qp=solver(); m=qp._build(d)
    assert m.Q.shape==(2,24,24) and m.A.shape==(2,0,24)
    assert torch.linalg.eigvalsh(m.Q).min()>0
    x=torch.randn(2,24,dtype=torch.float64,requires_grad=True)
    a=(m.acceleration_map@x[...,None]).squeeze(-1)+m.acceleration_offset
    applied=torch.cat((torch.zeros(2,6),x[:,:12]),-1)
    applied+=torch.einsum("bfkn,bfk->bn",d["foot_jacobians"],x[:,12:].reshape(2,4,3))
    applied+=torch.einsum("bkn,bk->bn",d["base_jacobian"],d["wrench_pred_world"])
    torch.testing.assert_close((d["mass_matrix"]@a[...,None]).squeeze(-1)+d["bias"],applied,atol=2e-14,rtol=2e-13)
    standard=.5*torch.einsum("bi,bij,bj->b",x/m.variable_scale,m.Q,x/m.variable_scale)+(m.p*x/m.variable_scale).sum(-1)
    direct=objective(qp,d,m,x)
    # Constants in affine squared residuals do not enter Q or p.
    torch.testing.assert_close(torch.autograd.grad(standard.sum(),x)[0],torch.autograd.grad(direct.sum(),x)[0],atol=1e-13,rtol=1e-12)


@pytest.mark.parametrize("pattern",range(16))
def test_swing_zero_nonredundant_constraints_and_native_pack(pattern):
    d=inputs(1); d["contact_probability"][0]=torch.tensor([(pattern>>i)&1 for i in range(4)])
    qp=solver(); m=qp._build(d)
    ns=pattern.bit_count()
    assert m.G.shape==(1,48+5*ns,24)
    assert m.A.shape==(1,3*(4-ns),24)
    if m.A.shape[1]: assert torch.linalg.matrix_rank(m.A)==m.A.shape[1]
    g,h,lo,hi=qp._cupiqp_native_pack(m)
    torch.testing.assert_close(g,m.G[:,24:]);torch.testing.assert_close(h,m.h[:,24:])
    torch.testing.assert_close(lo[:,:12],m.tau_lower/m.variable_scale[:12])
    assert torch.isneginf(lo[:,12:]).all() and torch.isposinf(hi[:,12:]).all()
    r=qp.solve(**d)
    assert r.stage.eq(0).all(),r.diagnostics
    assert torch.equal(r.force_world[d["contact_probability"]<.5],torch.zeros_like(r.force_world[d["contact_probability"]<.5]))
    x=torch.cat((r.tau_safe,r.force_world.flatten(1)),1)
    assert qp._certificate(m,x/m.variable_scale,1e-6)[0].all()


@pytest.mark.parametrize("beta",[.5,1.])
def test_joint_specific_intersections_and_exact_torque_rate(beta):
    qp=solver(position_integration_coefficient=beta,joint_acceleration_limits_rad_s2=tuple(range(10,22)))
    d=inputs();d["tau_nom"].fill_(100); d["joint_position"][:,0]=1.999
    d["joint_velocity"][:,1]=29.99
    m=qp._build(d)
    expected=torch.minimum(torch.arange(10,22,dtype=torch.float64),
        torch.minimum((30-d["joint_velocity"])/d["dt"],(2-d["joint_position"]-d["dt"]*d["joint_velocity"])/(beta*d["dt"].square())))
    torch.testing.assert_close(m.qdd_upper,expected)
    # Keep the velocity test inside position limits for actual feasibility.
    d["joint_position"][:,1]=-1.
    for _ in range(4):
        r=qp.solve(**d); assert r.stage.eq(0).all()
        assert ((r.tau_safe-d["previous_torque"]).abs()<=10).all()
        assert (r.tau_safe.abs()<=23.5).all()
        assert ((r.qdd[:,6:]<=qp._build(d).qdd_upper+1e-5)).all()
        d["previous_torque"]=r.tau_safe.detach()


def test_mixed_failures_bypass_solver_and_have_zero_gradients():
    d=inputs(5); d["mass_matrix"][1]=0; d["joint_position"][2]=10
    d["previous_torque"][3]=100; d["force_pred_world"][4,0,0]=float("nan")
    d["tau_nom"].requires_grad_();d["wrench_pred_world"].requires_grad_()
    qp=solver()
    with mock.patch.object(qp,"_backend_solve",wraps=qp._backend_solve) as call:
        r=qp.solve(**d)
        assert sum(c.args[0].p.shape[0] for c in call.call_args_list)==1
    assert r.stage.tolist()==[0,2,2,2,2]
    assert r.diagnostics["failure/mechanics"][1]
    assert r.diagnostics["failure/empty_qdd_intersection"][2]
    assert r.diagnostics["failure/empty_torque_intersection"][3]
    assert r.diagnostics["failure/nonfinite_input"][4]
    r.tau_safe.sum().backward()
    assert torch.isfinite(d["tau_nom"].grad).all()
    assert d["tau_nom"].grad[1:].eq(0).all()
    assert d["wrench_pred_world"].grad[1:].eq(0).all()


def test_exception_and_uncertified_rows_have_no_implicit_vjp():
    qp=solver(); d=inputs();d["tau_nom"].requires_grad_()
    with mock.patch.object(qp,"_backend_solve",side_effect=RuntimeError("forced")):
        r=qp.solve(**d)
    assert r.stage.eq(2).all() and not r.differentiated_mask.any()
    r.tau_safe.sum().backward();assert d["tau_nom"].grad.eq(0).all()
    # A finite but infeasible candidate must be rejected after exact clamp.
    with mock.patch.object(qp,"_backend_solve",return_value=QPBackendResult(torch.full((2,24),-1.,dtype=torch.float64))):
        r=qp.solve(**d)
    assert r.stage.eq(2).all()


def test_detachment_and_finite_difference_gradients():
    d=inputs(1);d["foot_jacobians"][:,:,:,6:]=torch.eye(12).reshape(4,3,12)*.3
    d["base_jacobian"][:,:,6:12]=torch.eye(6)*.2
    for name,t in d.items():
        if name!="dt":t.requires_grad_()
    qp=solver()
    r=qp.solve(**d); assert r.stage.eq(0).all()
    loss=r.tau_safe.square().sum()+.01*r.force_world.square().sum()
    loss.backward()
    for key in ("tau_nom","force_pred_world","wrench_pred_world"):
        assert torch.isfinite(d[key].grad).all() and d[key].grad.abs().sum()>0
        index=0 if key!="force_pred_world" else 2
        delta=1e-4; results=[]
        for sign in (-1,1):
            v={k:t.detach().clone() for k,t in d.items()}
            v[key].view(-1)[index]+=sign*delta
            out=qp.solve(**v)
            results.append(out.tau_safe.square().sum()+.01*out.force_world.square().sum())
        fd=(results[1]-results[0])/(2*delta)
        torch.testing.assert_close(d[key].grad.flatten()[index],fd,rtol=3e-3,atol=2e-5)
    for key in ("mass_matrix","bias","foot_jacobians","base_jacobian","contact_probability","joint_position","previous_torque","base_quaternion"):
        assert d[key].grad is None or d[key].grad.eq(0).all(),key


def test_changing_mechanics_refreshes_hessian_and_inference_has_no_graph():
    qp=solver();d=inputs();d["foot_jacobians"][:,:,:,6:]=torch.eye(12).reshape(4,3,12)
    first=qp._build(d);d["mass_matrix"]*=2;second=qp._build(d)
    assert not torch.equal(first.Q,second.Q)
    d["tau_nom"].requires_grad_()
    r=qp.solve(differentiable=False,**d)
    assert not r.tau_safe.requires_grad
    with torch.inference_mode(): qp.solve(differentiable=False,**d)
    qp.solve(**d).tau_safe.sum().backward()
    assert torch.isfinite(d["tau_nom"].grad).all()


def test_attitude_restores_tilt_with_physical_angular_acceleration():
    d=inputs(1);phi=torch.tensor(.2,dtype=torch.float64)
    d["base_quaternion"][0]=torch.tensor([torch.sin(phi/2),0,0,torch.cos(phi/2)])
    d["mass_matrix"][:,3,6]=.3;d["mass_matrix"][:,6,3]=.3
    d["tau_nom"].zero_()
    qp=solver(attitude_weight=100.,torque_tracking_weight=.01,contact_acceleration_weight=0.)
    r=qp.solve(**d); assert r.stage.eq(0).all()
    assert r.qdd[0,3]<0  # positive roll -> restoring negative roll acceleration


def test_qpth_warm_reference_reset_and_changed_settings():
    d=inputs();qp=solver(qpth_warm_start=True)
    expected=solver().solve(differentiable=False,**d)
    for _ in range(2):
        actual=qp.solve(differentiable=False,**d)
        torch.testing.assert_close(actual.tau_safe,expected.tau_safe,atol=1e-7,rtol=1e-7)
    qp.clear_warm_start(torch.tensor([0]))
    for _,owners,valid in qp._qpth_warm_states.values():
        assert not valid[owners==0].any()
    qp.cfg=replace(qp.cfg,force_scale_n=100.)
    m=qp._build(d);assert m.variable_scale[12:].eq(100.).all()


def test_diagnostics_do_not_change_primal_or_gradients():
    outputs=[]
    for level in ("minimal","physical","full"):
        d=inputs();d["tau_nom"].requires_grad_()
        qp=solver(diagnostics_level=level,full_audit_period=1,full_audit_sample_size=1)
        r=qp.solve(**d);r.tau_safe.sum().backward()
        outputs.append((r.tau_safe.detach(),d["tau_nom"].grad))
        if level=="minimal": assert not any(k.startswith("physical/") for k in r.diagnostics)
        else: assert r.diagnostics["physical/dynamics/joint_mae"].max()<1e-12
        if level=="full": assert torch.isfinite(r.diagnostics["full/audit/q_min_eigenvalue"]).sum()==1
    for x in outputs[1:]:
        for actual,expected in zip(x,outputs[0]):torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_cuda_cupiqp_canonical_native_forward_and_vjp(dtype):
    d=inputs(4,dtype,"cuda");d["contact_probability"][2:,:2]=0
    d["tau_nom"].requires_grad_();d["force_pred_world"].requires_grad_()
    qp=solver(qp_solver="cupiqp",solver_dtype=str(dtype).split(".")[-1])
    out=qp.solve(**d); assert out.stage.eq(0).all(),out.diagnostics
    loss=out.tau_safe.sum()+out.force_world.sum()*.01
    loss.backward(); assert torch.isfinite(d["tau_nom"].grad).all()
    assert d["tau_nom"].grad.abs().sum()>0
    ref=solver(solver_dtype="float64").solve(**{k:v.detach().double() for k,v in d.items()})
    assert ref.stage.eq(0).all()
    torch.testing.assert_close(out.tau_safe.double(),ref.tau_safe,rtol=1e-4,atol=2e-4)
    torch.testing.assert_close(out.force_world.double(),ref.force_world,rtol=1e-3,atol=2e-3)
