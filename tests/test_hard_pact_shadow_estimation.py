import torch
from scripts.hard_pact_shadow_estimation import acceleration, estimate


def test_rotating_frame_transport_and_joint_difference():
    q=torch.tensor([[0.,0.,0.,1.]],dtype=torch.double)
    v=torch.zeros(1,18,dtype=torch.double);v[:,0]=2;v[:,5]=3
    after=v.clone();after[:,6]=.04
    a=acceleration(q,v,after,.02)
    assert torch.allclose(a[0,:6],torch.tensor([0.,-6.,0.,0.,0.,0.],dtype=torch.double))
    assert a[0,6]==2


def test_coupled_force_sign_and_base_origin():
    torch.manual_seed(4)
    n=3;dtype=torch.double
    mass=torch.eye(18,dtype=dtype).expand(n,18,18)
    jt=torch.randn(n,18,12,dtype=dtype)*.1
    jt[:,6:]+=torch.eye(12,dtype=dtype)
    jb=torch.zeros(n,6,18,dtype=dtype);jb[:,:,:6]=torch.eye(6,dtype=dtype)
    # A force at an offset appears exactly once in the base-origin moment label.
    force=torch.tensor([[2.,0.,-10.]],dtype=dtype)
    arm=torch.tensor([[.1,0.,0.]],dtype=dtype)
    w=torch.cat((force,torch.cross(arm,force,dim=-1)),-1).expand(n,6)
    f=torch.randn(n,12,dtype=dtype)
    bias=(jt@f[...,None]).squeeze(-1);bias[:,:6]+=w
    out=estimate(mass,bias,jt.transpose(-1,-2).reshape(n,4,3,18),jb,
        torch.zeros(n,18,dtype=dtype),torch.zeros(n,12,dtype=dtype),torch.ones(n,4,dtype=dtype),1e-10)
    assert out['valid'].all()
    assert torch.allclose(out['force'].flatten(1),f,atol=1e-8)
    assert torch.allclose(out['wrench'],w,atol=1e-8)
    assert not out['force'].requires_grad


def test_swing_gates_and_nonfinite_rows():
    m=torch.eye(18).repeat(2,1,1);h=torch.zeros(2,18)
    j=torch.zeros(2,4,3,18);b=torch.zeros(2,6,18);b[:,:,:6]=torch.eye(6)
    a=torch.zeros(2,18);a[1,0]=float('nan')
    out=estimate(m,h,j,b,a,torch.zeros(2,12),torch.zeros(2,4))
    assert out['valid'].tolist()==[True,False]
    assert (out['force'][0]==0).all() and out['rank'][0]==0


def test_rotated_axes_and_finite_difference_cadence():
    q=torch.tensor([[0.,0.,2**-.5,2**-.5]],dtype=torch.double)
    before=torch.zeros(1,18,dtype=torch.double);after=before.clone();after[:,0]=.02
    a=acceleration(q,before,after,.02)
    assert torch.allclose(a[0,:3],torch.tensor([0.,-1.,0.],dtype=torch.double),atol=1e-12)
    # A held 20ms estimate must not be differentiated using 5ms.
    assert torch.allclose(acceleration(q,before,after,.005),a*4)


def test_reset_and_teleport_interval_exclusion():
    import numpy as np
    from scripts.report_hard_pact_shadow import valid_intervals
    a=dict(control_reset=np.zeros((4,2),bool),control_reasons=np.zeros((4,2,7),bool),
        shadow_position_before=np.zeros((16,2,3)),shadow_position_after=np.zeros((16,2,3)))
    a['control_reset'][1,0]=True;a['control_reasons'][2,1,6]=True
    a['shadow_position_after'][12,0,0]=2
    valid=valid_intervals(a)
    assert not valid[:4].any() and not valid[4:8,0].any() and not valid[8:12,1].any()
    assert not valid[12,0] and valid[13:,0].all()
