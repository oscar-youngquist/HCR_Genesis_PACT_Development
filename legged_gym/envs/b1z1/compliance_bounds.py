"""Batched first exit of a Cartesian segment from a spherical workspace.

All intersections with radius spheres, elevation cones and azimuth planes
are enumerated. Classifying the intervening intervals prevents tunnelling
through the inner-radius hole, including when both endpoints are admissible.
"""
import torch


def feasible(position, center, bounds):
    from .directional_compliance import chart
    s=chart(position-center,bounds)
    return torch.isfinite(s).all(-1)&((s>=bounds[:,0]-1e-6)&(s<=bounds[:,1]+1e-6)).all(-1)


def first_exit(start, end, center, bounds):
    from .directional_compliance import chart
    p, d = start-center, end-start
    a = d.square().sum(-1); b = 2*(p*d).sum(-1)
    c = p.square().sum(-1)
    angles = bounds[1]
    weights = torch.stack((-angles.sin().square(), -angles.sin().square(), angles.cos().square()), -1)
    aa = torch.cat((a[..., None].expand(*a.shape, 2), (d.square()[..., None, :]*weights).sum(-1)), -1)
    bb = torch.cat((b[..., None].expand(*b.shape, 2), (2*p*d)[..., None, :].mul(weights).sum(-1)), -1)
    cc = torch.cat((c[..., None]-bounds[0].square(), (p.square()[..., None, :]*weights).sum(-1)), -1)
    disc = bb.square()-4*aa*cc
    quadratic = aa.abs() > 1e-14
    denom = torch.where(quadratic, 2*aa, torch.ones_like(aa))
    root = disc.clamp_min(0).sqrt()
    roots = torch.stack(((-bb-root)/denom, (-bb+root)/denom), -1)
    roots = torch.where((quadratic & (disc >= 0))[..., None], roots, torch.ones_like(roots))
    linear = torch.where(bb.abs()>1e-14, -cc/torch.where(bb.abs()>1e-14,bb,torch.ones_like(bb)), torch.ones_like(cc))
    roots[..., 0] = torch.where(~quadratic,linear,roots[..., 0])
    y = bounds[2]
    az_a = d[..., 1:2]*y.cos()-d[..., 0:1]*y.sin()
    az_b = p[..., 1:2]*y.cos()-p[..., 0:1]*y.sin()
    az = torch.where(az_a.abs()>1e-14,-az_b/torch.where(az_a.abs()>1e-14,az_a,torch.ones_like(az_a)),torch.ones_like(az_a))
    times = torch.cat((roots.flatten(-2),az,torch.zeros_like(a[..., None]),torch.ones_like(a[..., None])), -1)
    times = torch.where((times>=0)&(times<=1),times,torch.ones_like(times)).sort(-1).values
    mid=(times[..., :-1]+times[..., 1:])/2
    s=chart(p[..., None, :]+mid[..., None]*d[..., None, :],bounds)
    valid=((s>=bounds[:,0]-1e-7)&(s<=bounds[:,1]+1e-7)).all(-1)
    invalid=(~valid)&((times[..., 1:]-times[..., :-1])>1e-9)
    exit_time=torch.where(invalid,times[..., :-1],torch.ones_like(mid)).amin(-1)
    # Stay just inside a first crossing despite float32 intersection roundoff.
    return torch.where(exit_time<1,(exit_time-1e-6).clamp_min(0),exit_time)


def bound_reference(position, velocity, center, bounds, start=None):
    from .directional_compliance import chart, sphere_basis
    if start is None:
        start=position  # Velocity-only compatibility call at an accepted point.
    alpha=first_exit(start,position,center,bounds)
    accepted=start+alpha[...,None]*(position-start)
    s=chart(accepted-center,bounds)
    rates=(sphere_basis(s)*velocity.unsqueeze(-2)).sum(-1)
    outward=(((s<=bounds[:,0]+2e-6)&(rates<0))|((s>=bounds[:,1]-2e-6)&(rates>0))).any(-1)
    # Stop, never rotate/project velocity into a disabled Cartesian component.
    stopped=(alpha<1)|outward
    return accepted,torch.where(stopped[...,None],torch.zeros_like(velocity),velocity)
