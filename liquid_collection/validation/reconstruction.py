"""Compact-only offline reconstruction. No native reference is accepted here.

Hermite x and normalized quaternion Hermite q use the two surrounding saved
endpoints (offline future endpoint). Their derivatives define u, omega, a, alpha.
Recorded r,v,S have independent C2 cubic interpolants. World held inputs are
right-continuous event values. Wrench is the average on its OWN (start,end]
interval, held in world coordinates then shifted/rotated at query time. It is
not asserted to be an instantaneous truth or an online causal interpolator.
"""
import numpy as np
import torch
from scipy.interpolate import CubicHermiteSpline, CubicSpline

T=torch.float64

def tensor(x): return torch.as_tensor(np.asarray(x),dtype=T)
def cross(a,b): return torch.linalg.cross(a,b)
def skew(v):
    z=torch.zeros_like(v[...,0]); x,y,w=v.unbind(-1)
    return torch.stack((z,-w,y,w,z,-x,-y,x,z),-1).reshape(v.shape[:-1]+(3,3))
def matvec(R,v): return (R@v[...,None])[...,0]
def vee(A): return torch.stack((A[...,2,1]-A[...,1,2],A[...,0,2]-A[...,2,0],A[...,1,0]-A[...,0,1]),-1)/2

def rotation_derivatives(q,dq,ddq):
    n=torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    nd=(q*dq).sum(-1,keepdim=True)/n
    ndd=((dq*dq+q*ddq).sum(-1,keepdim=True)-nd*nd)/n
    p=q/n; pd=dq/n-q*nd/n**2
    pdd=ddq/n-2*dq*nd/n**2-q*ndd/n**2+2*q*nd**2/n**3
    def bilinear(a,b):
        aw,av=a[...,:1],a[...,1:]; bw,bv=b[...,:1],b[...,1:]
        I=torch.eye(3,dtype=T)
        return ((aw*bw-(av*bv).sum(-1,keepdim=True))[...,None]*I+
                2*av[..., :,None]*bv[...,None,:]+2*aw[...,None]*skew(bv))
    R=bilinear(p,p); Rd=bilinear(pd,p)+bilinear(p,pd)
    Rdd=bilinear(pdd,p)+2*bilinear(pd,pd)+bilinear(p,pdd)
    return R,Rd,Rdd

class Polynomial:
    def __init__(self,t,y,dy=None):
        spline=CubicSpline(t,y) if dy is None else CubicHermiteSpline(t,y,dy)
        self.t=tensor(t); self.c=tensor(spline.c)
    def __call__(self,t,derivative=0):
        i=torch.searchsorted(self.t,t.contiguous(),right=True)-1
        i=i.clamp(0,len(self.t)-2); x=(t-self.t[i])[...,None]
        c=self.c[:,i]
        if derivative==0: return ((c[0]*x+c[1])*x+c[2])*x+c[3]
        if derivative==1: return (3*c[0]*x+2*c[1])*x+c[2]
        return 6*c[0]*x+2*c[1]

class Reconstruction:
    def __init__(self, states, events, intervals):
        # Inputs are single-environment, single-episode rows, already isolated.
        self.times=np.array([r['time_s'] for r in states]); self.start=self.times[0]; self.end=self.times[-1]
        def field(k): return np.stack([r[k] for r in states])
        self.x=Polynomial(self.times,field('x_O_W'),field('velocity_W')[:,:3])
        q=field('q_WB_wxyz').astype(float)
        for i in range(1,len(q)):
            if np.dot(q[i-1],q[i])<0: q[i]*=-1
        w=field('velocity_W')[:,3:]
        dq=.5*np.concatenate((-np.sum(w*q[:,1:],axis=-1,keepdims=True), q[:,:1]*w+np.cross(w,q[:,1:])),axis=-1)
        self.q=Polynomial(self.times,q,dq)
        self.aggregates={k:Polynomial(self.times,field(k)) for k in ('r_B','v_B','S_B')}
        self.mass=float(states[0]['mass']); self.g=tensor(states[0]['gravity_W'])
        self.et=tensor([e['time_s'] for e in events]); self.held=tensor([e['held_acceleration_W'] for e in events])
        self.it=tensor([r['time_s'] for r in intervals]); self.duration=tensor([r['interval_duration_s'] for r in intervals])
        self.wrench=tensor([np.r_[r['coupling_force_impulse_W'],r['coupling_moment_impulse_world_origin_W']]/r['interval_duration_s'] for r in intervals])
    def evaluate(self,t,reference='interval'):
        t=t if isinstance(t,torch.Tensor) else tensor(t)
        if bool(((t<self.start-1e-9)|(t>self.end+1e-9)).any()): raise ValueError('No extrapolation')
        x,u,a=self.x(t),self.x(t,1),self.x(t,2)
        R,Rd,Rdd=rotation_derivatives(self.q(t),self.q(t,1),self.q(t,2)); Rt=R.transpose(-1,-2)
        omegaW=vee(Rd@Rt); alphaW=vee(Rdd@Rt+Rd@Rd.transpose(-1,-2))
        omega=matvec(Rt,omegaW); alpha=matvec(Rt,alphaW)
        uB=matvec(Rt,u); aB=matvec(Rt,a); gB=matvec(Rt,self.g.expand_as(u))
        ei=(torch.searchsorted(self.et,t.contiguous(),right=True)-1).clamp(0,len(self.et)-1)
        h=self.held[ei]; hlin=matvec(Rt,h[...,:3]); hang=matvec(Rt,h[...,3:])
        z=torch.cat((gB,uB,omega,hlin,hang),-1)
        zd=torch.cat((-cross(omega,gB),aB-cross(omega,uB),alpha,-cross(omega,hlin),-cross(omega,hang)),-1)
        wi=torch.searchsorted(self.it,t.contiguous(),right=False).clamp(0,len(self.it)-1)
        # Native-reference sensitivity only; callers never pass native data to compact reconstruction.
        if reference=='midpoint_linear':
            mid=self.it-self.duration/2
            j=torch.searchsorted(mid,t.contiguous()).clamp(1,len(mid)-1)
            f=((t-mid[j-1])/(mid[j]-mid[j-1])).clamp(0,1)[...,None]
            wrench=self.wrench[j-1]*(1-f)+self.wrench[j]*f
        else: wrench=self.wrench[wi]
        F=matvec(Rt,wrench[...,:3]); tau=matvec(Rt,wrench[...,3:]-cross(x,wrench[...,:3]))
        vals={k:self.aggregates[k](t) for k in self.aggregates}
        r,v,S=(vals[k] for k in ('r_B','v_B','S_B'))
        rd,vd,Sd=(self.aggregates[k](t,1) for k in ('r_B','v_B','S_B'))
        A=aB+cross(alpha,r)+cross(omega,cross(omega,r))+cross(omega,rd+v)+vd
        return dict(x=x,R=R,u=u,omega=omega,a=a,alpha=alpha,z=z,zdot=zd,F=F,tau=tau,
            r=r,v=v,S=S,rdot=rd,vdot=vd,Sdot=Sd,
            e_kin=rd-v,e_Newton=self.mass*(A-gB)+F,
            e_Euler=Sd+cross(omega,S)+tau-cross(r,F),
            P=self.mass*(u+matvec(R,v+cross(omega,r))),SW=matvec(R,S),
            dP=self.mass*self.g-matvec(R,F),dS=matvec(R,-tau+cross(r,F)))
