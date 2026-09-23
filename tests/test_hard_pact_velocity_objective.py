"""Soft body-velocity QP objective: algebra, replay, frames and gradients."""
from dataclasses import replace
import pytest
import torch
from test_hard_pact_reduced_qp import inputs, solver
from test_hard_pact_qp_modes import fixture


def data(n=2):
    d=inputs(n)
    d['base_linear_velocity_world']=torch.tensor([.2,-.1,.3],dtype=torch.float64).expand(n,-1).clone()
    d['velocity_command']=torch.tensor([.6,.1,.2],dtype=torch.float64).expand(n,-1).clone()
    d['foot_jacobians'][:,:,:,:3]=torch.eye(3,dtype=torch.float64)*.1
    d['foot_jacobians'][:,:,2,5]=.02
    # SPD base/joint coupling makes torque influence base acceleration.
    d['mass_matrix'][:,0,6]=d['mass_matrix'][:,6,0]=.1
    return d


def test_zero_weight_parity_and_required_replay():
    q=solver();d=inputs()
    a=q._build(d);d.update(base_linear_velocity_world=torch.full((2,3),float('nan')),
                          velocity_command=torch.full((2,3),float('nan')))
    b=q._build(d)
    for name in ('Q','p','G','h'):
        assert torch.equal(getattr(a,name),getattr(b,name))
    q.cfg=replace(q.cfg,planar_velocity_weight=1.)
    with pytest.raises(ValueError,match='captured physical'):
        q.solve(**inputs())


def test_affine_objective_psd_scaling_and_recovery():
    q=solver();d=data();base=q._build(d)
    q.cfg=replace(q.cfg,planar_velocity_weight=2.,yaw_rate_weight=3.,
                  planar_velocity_scale_m_s=.7,yaw_rate_scale_rad_s=1.5)
    m=q._build(d)
    C,e,_=q._velocity_tracking_affine(d,m.acceleration_map,m.acceleration_offset)
    weights=C.new_tensor([2/.7**2,2/.7**2,3/1.5**2])
    Cz=C*m.variable_scale
    expected_Q=2*Cz.transpose(1,2)@(weights[None,:,None]*Cz)
    expected_p=2*(Cz.transpose(1,2)@(weights*e)[...,None]).squeeze(-1)
    torch.testing.assert_close(m.Q-base.Q,expected_Q,atol=1e-13,rtol=1e-10)
    torch.testing.assert_close(m.p-base.p,expected_p,atol=1e-13,rtol=1e-10)
    assert torch.linalg.eigvalsh(expected_Q).min() > -1e-12
    recovery=q._soft_joint_problem(m)
    torch.testing.assert_close(recovery.Q[:,:24,:24],m.Q)
    torch.testing.assert_close(recovery.p[:,:24],m.p)
    assert torch.equal(m.G,base.G) and torch.equal(m.h,base.h)


def test_rotating_body_bias_cancellation():
    q=solver();d=data(1)
    # Nonidentity roll/pitch/yaw rotation, nonzero linear and angular velocity.
    axis=torch.tensor([.2,.4,.7],dtype=torch.float64);axis=axis/axis.norm()
    angle=torch.tensor(.8,dtype=torch.float64)
    quat=torch.cat((axis*torch.sin(angle/2),torch.cos(angle/2).view(1)))
    d['base_quaternion']=quat[None]
    from legged_gym.dynamics.bard_go2_dynamics import _quat_wxyz_rotation
    R=_quat_wxyz_rotation(quat[[3,0,1,2]][None])[0]
    v=torch.tensor([.3,-.2,.7],dtype=torch.float64)
    w=torch.tensor([.4,.5,-.3],dtype=torch.float64)
    d['base_linear_velocity_world']=(R@v)[None]
    d['base_angular_velocity_world']=(R@w)[None]
    amap=torch.eye(18,dtype=torch.float64)[None];offset=torch.zeros(1,18,dtype=torch.float64)
    C,e,now=q._velocity_tracking_affine(d,amap,offset)
    torch.testing.assert_close(now,torch.cat((v[:2],w[2:]))[None])
    a=torch.arange(18,dtype=torch.float64)/100
    # Classical world acceleration includes transport; transform back and
    # subtract rotating-frame derivative. Omitting either term fails here.
    classical=R@(a[:3]+torch.cross(w,v,dim=0))
    body_derivative=R.T@classical-torch.cross(w,v,dim=0)
    expected=now+d['dt']*torch.cat((body_derivative[:2],a[5:6]))
    torch.testing.assert_close((C@a[None,:,None]).squeeze(-1)+e+d['velocity_command'],expected)


def test_gated_diagnostics_recovery_and_cache_preservation(monkeypatch):
    q=solver(planar_velocity_weight=1.,yaw_rate_weight=1.,diagnostics_level='physical')
    d=data();d['joint_position'][1].fill_(2.01)
    instances=q._backend_instances
    monkeypatch.setattr(q,'clear_warm_start',lambda:pytest.fail('numeric objective invalidated caches'))
    q.cfg=replace(q.cfg,planar_velocity_weight=2.,yaw_rate_weight=3.)
    result=q.solve(differentiable=False,**d)
    assert q._backend_instances is instances
    assert result.stage.tolist()==[0,1]
    metrics=q.iteration_metrics('rollout',d['tau_nom'])
    keys=[k for k in metrics if 'model_velocity_tracking' in k]
    assert any('/primary/accepted/full_candidate/' in k for k in keys)
    assert any('/recovery/accepted/full_candidate/' in k for k in keys)
    assert all(torch.isfinite(metrics[k]) for k in keys if '/accepted/' in k and not k.endswith('count'))


@pytest.mark.parametrize('mode',['every_substep','random_one_substep'])
def test_exact_command_sampling_and_shuffled_replay(mode):
    task,_,qp,_,_,q,v,quat,_=fixture(mode)
    qp.cfg=replace(qp.cfg,planar_velocity_weight=1.,yaw_rate_weight=1.)
    task.commands=torch.zeros(8,3)
    task._begin_qp_interval()
    for k in range(4):
        task.commands[:]=torch.arange(8)[:,None]+k*.1
        task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(8,6))
    packet=task._qp_sampled_transition
    expected=(torch.arange(8)+packet['sampled_qp_substep_index'].flatten()*.1)[:,None].expand(-1,3)
    order=torch.tensor([3,0,5,1,7,4,2,6])
    task.commands.fill_(999)
    torch.testing.assert_close(packet['sampled_qp_velocity_command'][order],expected[order])


@pytest.mark.parametrize('backend',['qpth','cupiqp'])
def test_tracking_vjp_and_detached_state(backend):
    if backend == 'cupiqp' and not torch.cuda.is_available():
        pytest.skip('cuPIQP requires CUDA')
    # Resolve explicitly to float64 for a 1e-4 central-difference probe; CUDA
    # auto intentionally chooses float32 and cannot resolve this subtraction.
    q=solver(planar_velocity_weight=30.,yaw_rate_weight=20.,qp_solver=backend,solver_dtype='float64')
    d=data(1)
    if backend == 'cupiqp':
        d={key:value.cuda() for key,value in d.items()}
    for name in ('tau_nom','force_pred_world','wrench_pred_world','velocity_command',
                 'mass_matrix','base_linear_velocity_world','base_quaternion'):
        d[name].requires_grad_()
    def output(values):
        r=q.solve(differentiable=True,**values)
        assert r.differentiated_mask.all()
        return r.tau_safe.sum()+r.force_world.sum()*.01
    value=output(d)
    names=['tau_nom','force_pred_world','wrench_pred_world']
    grads=torch.autograd.grad(value,[d[n] for n in names]+[d['velocity_command'],d['mass_matrix'],
        d['base_linear_velocity_world'],d['base_quaternion']],allow_unused=True)
    assert all(g is None for g in grads[3:])
    for name,grad in zip(names,grads):
        direction=torch.ones_like(d[name])*.1
        eps=1e-4
        with torch.no_grad():
            hi=output(dict(d,**{name:d[name]+eps*direction}))
            lo=output(dict(d,**{name:d[name]-eps*direction}))
        torch.testing.assert_close((grad*direction).sum(),(hi-lo)/(2*eps),rtol=2e-3,atol=2e-5)
        assert torch.isfinite(grad).all() and grad.abs().sum()>0
