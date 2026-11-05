"""Analytic EE sphere distances; no arm-arm tests or spacing inflation."""
import torch


def point_segment(p, a, b):
    d=b-a
    u=((p-a)*d).sum(-1)/d.square().sum(-1).clamp_min(1e-15)
    return (p-(a+u.clamp(0,1)[...,None]*d)).norm(dim=-1)


def segment_segment(p, q, a, b):
    u,v,w=q-p,b-a,p-a
    aa,bb,cc=(u*u).sum(-1),(u*v).sum(-1),(v*v).sum(-1)
    dd,ee=(u*w).sum(-1),(v*w).sum(-1)
    det=aa*cc-bb.square()
    denominator=torch.where(det>1e-15,det,torch.ones_like(det))
    s,t=(bb*ee-cc*dd)/denominator,(aa*ee-bb*dd)/denominator
    interior=(w+s[...,None]*u-t[...,None]*v).norm(dim=-1)
    interior=torch.where((det>1e-15)&(s>=0)&(s<=1)&(t>=0)&(t<=1),interior,torch.full_like(interior,float('inf')))
    return torch.stack((point_segment(p,a,b),point_segment(q,a,b),
                        point_segment(a,p,q),point_segment(b,p,q),interior),-1).amin(-1)


def point_box(p, center, half):
    d=(p-center).abs()-half
    return d.clamp_min(0).norm(dim=-1)+d.amax(-1).clamp_max(0)


def segment_box(p, q, center, half):
    # Squared distance to an AABB is quadratic between face crossings. Solve
    # its minimum on every interval, rather than inflating sampled segments.
    p=p-center;v=q-center-p
    denom=torch.where(v.abs()>1e-15,v,torch.ones_like(v))
    roots=torch.stack(((-half-p)/denom,(half-p)/denom),-1)
    roots=torch.where((v.abs()>1e-15)[...,None],roots,torch.ones_like(roots))
    roots=roots.flatten(-2).clamp(0,1)
    one=torch.ones_like(roots[...,:1]);zero=torch.zeros_like(one)
    times=torch.cat((zero,roots,one),-1).sort(-1).values
    mid=(times[...,:-1]+times[...,1:])/2
    x=p[...,None,:]+mid[...,None]*v[...,None,:]
    outside=x.abs()>half
    offset=p[...,None,:]-x.sign()*half
    a=(outside*v[...,None,:].square()).sum(-1)
    b=(outside*offset*v[...,None,:]).sum(-1)
    optimum=(-b/a.clamp_min(1e-15)).clamp(times[...,:-1],times[...,1:])
    distances=(p[...,None,:]+optimum[...,None]*v[...,None,:]).abs().sub(half).clamp_min(0).norm(dim=-1)
    unsigned=distances.amin(-1)
    # Inside the box the signed distance is the maximum of six affine face
    # distances. Its minimum lies at an endpoint or a pairwise intersection.
    # Keep this depth so crossing an obstacle is not misclassified as retreat.
    intercept=torch.cat((p-half,-p-half),-1)
    slope=torch.cat((v,-v),-1)
    i,j=torch.triu_indices(6,6,1,device=p.device)
    denominator=slope[...,i]-slope[...,j]
    crossings=(intercept[...,j]-intercept[...,i])/torch.where(denominator.abs()>1e-15,denominator,torch.ones_like(denominator))
    crossings=torch.where(denominator.abs()>1e-15,crossings,torch.zeros_like(crossings)).clamp(0,1)
    samples=torch.cat((zero,one,crossings),-1)
    depth=(intercept[...,None,:]+samples[...,None]*slope[...,None,:]).amax(-1).amin(-1)
    return torch.where(depth<0,depth,unsigned)
