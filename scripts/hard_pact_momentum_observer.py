"""Detached, evaluation-only centroidal momentum observer. No controller imports."""
import torch

DEFAULT_GAINS=(25.,25.,25.,25.,25.,20.)


def base_origin_wrench(wrench_com,com,base):
    return torch.cat((wrench_com[...,:3],wrench_com[...,3:]+torch.cross(
        com-base,wrench_com[...,:3],dim=-1)),-1)


def contact_input(mass,com,feet,forces):
    linear=forces.sum(-2).clone();linear[...,2]-=mass*9.81
    return torch.cat((linear,torch.cross(feet-com[...,None,:],forces,dim=-1).sum(-2)),-1)


class MomentumObserver:
    """Backward-Euler observer, using current H and the completed interval's u.

    Hhat[k]=(Hhat[k-1]+dt*(u[k-1,k]+K*H[k]))/(1+dt*K).
    W[k]=K*(H[k]-Hhat[k]). Stable for any positive dt; discrete pole 1/(1+K*dt).
    This integrates momentum, never differentiating velocity/gyro. W is available
    at k, not retroactively at k-1. A held W[k] is a separate next-interval forecast.
    Invalid/duplicate timestamps invalidate that row and require reinitialization.
    """
    def __init__(self,n,gains=DEFAULT_GAINS,device='cpu',dtype=torch.float64):
        self.k=torch.as_tensor(gains,device=device,dtype=dtype)
        if self.k.shape!=(6,) or not torch.isfinite(self.k).all() or not (self.k>0).all():
            raise ValueError('Six finite positive gains required')
        self.hat=torch.zeros(n,6,device=device,dtype=dtype)
        self.time=torch.zeros(n,device=device,dtype=dtype)
        self.age=torch.zeros_like(self.time);self.ready=torch.zeros(n,device=device,dtype=torch.bool)

    @torch.no_grad()
    def update(self,h,u,t,reset=None):
        t=torch.as_tensor(t,device=h.device,dtype=h.dtype).expand(h.shape[0])
        reset=torch.zeros_like(self.ready) if reset is None else reset
        dt=t-self.time
        finite=h.isfinite().all(-1)&u.isfinite().all(-1)&t.isfinite()
        start=(~self.ready|reset)&finite
        advance=self.ready&~reset&finite&(dt>0)
        safe_dt=torch.where(advance,dt,torch.zeros_like(dt))[:,None]
        proposed=(self.hat+safe_dt*(torch.nan_to_num(u)+self.k*torch.nan_to_num(h)))/(1+safe_dt*self.k)
        self.hat=torch.where(start[:,None],torch.nan_to_num(h),torch.where(advance[:,None],proposed,self.hat))
        self.age=torch.where(start,torch.zeros_like(dt),self.age+torch.where(advance,dt,torch.zeros_like(dt)))
        self.time=torch.where(start|advance,t,self.time);self.ready=start|advance
        w=self.k*(h-self.hat)
        return dict(wrench=w.masked_fill(~advance[:,None],float('nan')),available=advance,
            initialized=start,burnin=self.age<3/self.k.min(),age=self.age.clone())
