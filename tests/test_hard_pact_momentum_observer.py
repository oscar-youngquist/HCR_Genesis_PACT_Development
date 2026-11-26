import torch
import pytest
from scripts.hard_pact_momentum_observer import MomentumObserver,contact_input,base_origin_wrench


def test_static_gravity_support():
    com=torch.zeros(1,3,dtype=torch.double);feet=torch.tensor([[[-1.,0.,0.],[1.,0.,0.]]],dtype=torch.double)
    f=torch.tensor([[[0.,0.,9.81],[0.,0.,9.81]]],dtype=torch.double)
    u=contact_input(2.,com,feet,f);assert torch.allclose(u,torch.zeros_like(u))
    o=MomentumObserver(1)
    for t in range(100):r=o.update(torch.zeros_like(u),u,t*.005)
    assert torch.allclose(r['wrench'],u)


def test_each_axis_step_sign_response_and_variable_dt():
    for gain in (10.,25.,50.):
        o=MomentumObserver(6,gains=[gain]*6);h=torch.zeros(6,6,dtype=torch.double)
        u=h.clone();o.update(h,u,0.);time=0.;expected=torch.zeros_like(h);load=torch.eye(6,dtype=torch.double)*2
        for i in range(300):
            dt=.003 if i%2 else .007;time+=dt
            external=load if i>=20 else torch.zeros_like(load)
            # Independent Newton-Euler momentum integration under known load.
            h=h+dt*external
            expected=(expected+gain*dt*external)/(1+gain*dt)
            r=o.update(h,u,time)
            assert torch.allclose(r['wrench'],expected,atol=1e-11)
        assert torch.allclose(r['wrench'],load,atol=1e-5)


def test_moving_limbs_and_origin():
    # Two moving point limbs; zero torso angular velocity, nonzero whole-body L.
    p=torch.tensor([[-1.,0.,0.],[1.,0.,0.]],dtype=torch.double)
    v=torch.tensor([[0.,-2.,0.],[0.,2.,0.]],dtype=torch.double)
    momentum=torch.cat((v.sum(0),torch.cross(p,v,dim=-1).sum(0)))
    assert momentum[5]==4
    r=torch.tensor([[0.,1.,0.]],dtype=torch.double);w=torch.tensor([[2.,0.,0.,0.,0.,4.]],dtype=torch.double)
    assert torch.allclose(base_origin_wrench(w,r,torch.zeros_like(r)),torch.tensor([[2.,0.,0.,0.,0.,2.]],dtype=torch.double))


def test_independent_ballistic_point_masses_rotated():
    # Exact Newton translation of two masses in rotated coordinates, not observer recurrence.
    angle=.7;R=torch.tensor([[torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle)),0],
        [torch.sin(torch.tensor(angle)),torch.cos(torch.tensor(angle)),0],[0,0,1]],dtype=torch.double)
    masses=torch.tensor([2.,3.],dtype=torch.double);force=torch.tensor([3.,-2.,1.],dtype=torch.double)@R.T
    o=MomentumObserver(1);u=torch.zeros(1,6,dtype=torch.double)
    for i in range(501):
        t=i*.005;vel=(force/masses.sum()*t).expand(2,3)
        h=torch.cat(((masses[:,None]*vel).sum(0),torch.zeros(3,dtype=torch.double)))[None]
        out=o.update(h,u,t)
    assert torch.allclose(out['wrench'][0,:3],force,atol=1e-6)


def test_partial_reset_invalid_and_duplicate_order():
    o=MomentumObserver(2);h=torch.ones(2,6,dtype=torch.double);u=torch.zeros_like(h)
    o.update(h,u,0.);o.update(h*2,u,.01)
    r=o.update(h*3,u,.02,torch.tensor([True,False]));assert r['initialized'].tolist()==[True,False]
    assert r['available'].tolist()==[False,True]
    r=o.update(h*3,u,.02);assert not r['available'].any()
    r=o.update(h*4,u,.03);assert r['initialized'].all()
    h[0,0]=float('nan');r=o.update(h,u,.04);assert not r['available'][0] and r['available'][1]


def test_accelerating_two_limb_rotor_independent_newton_dynamics():
    # Two unit point masses, unit radius: imposed tangential forces give torque
    # 2*alpha; centripetal forces cancel. Momentum is independently summed.
    o=MomentumObserver(1);alpha=.3
    for i in range(601):
        t=i*.005;angle=.5*alpha*t*t
        pos=torch.tensor([[torch.cos(torch.tensor(angle)),torch.sin(torch.tensor(angle)),0]],dtype=torch.double)
        pos=torch.cat((pos,-pos));omega=torch.tensor([0.,0.,alpha*t],dtype=torch.double)
        velocity=torch.cross(omega.expand_as(pos),pos,dim=-1)
        h=torch.cat((velocity.sum(0),torch.cross(pos,velocity,dim=-1).sum(0)))[None]
        out=o.update(h,torch.zeros_like(h),t)
    assert abs(out['wrench'][0,5]-2*alpha)<1e-4
    assert out['wrench'][0,:5].abs().max()<1e-8


def test_invalid_gains_and_backward_timestamp():
    for gains in ([0]*6,[-1]*6,[float('nan')]*6,[1]*5):
        with pytest.raises(ValueError):MomentumObserver(1,gains)
    o=MomentumObserver(1);h=torch.zeros(1,6,dtype=torch.double)
    o.update(h,h,1.);assert not o.update(h,h,.9)['available'].any()
    assert o.update(h,h,1.1)['initialized'].all()
