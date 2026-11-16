"""FACET-only command-space geometry; no IK or physical force application.

Positive clearances point into the permitted region. Coordinates are relative
to the heading-aligned workspace center (world vertical), not world travel.
The box is the task's existing EE exclusion geometry, not a full-arm model.
"""
import torch


def geometry(q, nominal, cfg):
    r = q.norm(dim=-1).clamp_min(1e-8)
    rho = q[..., :2].norm(dim=-1).clamp_min(1e-8)
    radial = q / r[..., None]
    pitch = torch.atan2(q[..., 2], rho)
    yaw = torch.atan2(q[..., 1], q[..., 0])
    ep = torch.stack((-q[..., 0]*q[..., 2]/(rho*r),
                      -q[..., 1]*q[..., 2]/(rho*r), rho/r), -1)
    ey = torch.stack((-q[..., 1]/rho, q[..., 0]/rho, torch.zeros_like(r)), -1)
    delta = q-nominal
    length = delta.norm(dim=-1).clamp_min(1e-8)
    # Angular constraints expressed in metres using local arc lengths.
    distances = [r-cfg.radius[0], cfg.radius[1]-r,
        r*(pitch-cfg.pitch[0]), r*(cfg.pitch[1]-pitch),
        rho*(yaw-cfg.azimuth[0]), rho*(cfg.azimuth[1]-yaw),
        cfg.max_offset-length, q[..., 2]-cfg.ground]
    z = torch.zeros_like(q); z[..., 2] = 1
    normals = [radial, -radial, ep, -ep, ey, -ey, -delta/length[..., None], z]
    lo, hi = q.new_tensor(cfg.box_low), q.new_tensor(cfg.box_high)
    center, half = (lo+hi)/2, (hi-lo)/2
    d = q-center
    outside = d.abs()-half
    nearest = torch.maximum(torch.minimum(q, hi), lo)
    v = q-nearest
    norm = v.norm(dim=-1)
    axis = outside.argmax(-1)
    inside_normal = torch.nn.functional.one_hot(axis, 3).to(q)*d.sign()
    normal = torch.where((norm > 1e-8)[..., None], v/norm.clamp_min(1e-8)[..., None], inside_normal)
    distances.append(norm + outside.amax(-1).clamp_max(0))
    normals.append(normal)
    return torch.stack(distances, -1), torch.stack(normals, -2)


def restoring_force(q, nominal, velocity, cfg, nominal_velocity=None):
    distance, normal = geometry(q, nominal, cfg)
    # C1 force onset; bounded activation avoids abrupt stiffness jumps.
    penetration = (cfg.boundary_margin-distance).clamp_min(0)
    activation = (penetration/cfg.boundary_margin).clamp_max(1)
    normal_speed = (normal*velocity[..., None, :]).sum(-1)
    if nominal_velocity is not None:
        normal_speed[..., 6] -= (normal[..., 6, :]*nominal_velocity).sum(-1)
    outward = normal_speed.clamp_max(0)
    magnitude = cfg.boundary_stiffness*penetration*activation-cfg.boundary_damping*outward*activation
    return (normal*magnitude[..., None]).sum(-2)


def guard(q, nominal, velocity, previous, cfg, nominal_velocity=None):
    """Fixed geometric projections, then analytic segment/box interception.

    Velocity loses only components directed into an active boundary. This is
    not an IK feasibility certificate and does not constrain intermediate links.
    """
    lo, hi = q.new_tensor(cfg.box_low), q.new_tensor(cfg.box_high)
    travel = q-previous
    safe_den = torch.where(travel.abs() > 1e-12, travel, torch.ones_like(travel))
    a, b = (lo-previous)/safe_den, (hi-previous)/safe_den
    parallel = travel.abs() <= 1e-12
    near = torch.where(parallel, torch.full_like(a, -torch.inf), torch.minimum(a, b)).amax(-1)
    far = torch.where(parallel, torch.full_like(a, torch.inf), torch.maximum(a, b)).amin(-1)
    outside_parallel = (parallel & ((previous < lo) | (previous > hi))).any(-1)
    hit = (~outside_parallel) & (near >= 0) & (near <= 1) & (far >= near)
    q = torch.where(hit[..., None], previous+(near-1e-5).clamp_min(0)[..., None]*travel, q)
    for _ in range(4):  # Fixed-cost constraint projections, not iterative IK.
        distance, normals = geometry(q, nominal, cfg)
        worst, index = distance.min(-1)
        n = normals.gather(-2, index[..., None, None].expand(*index.shape, 1, 3)).squeeze(-2)
        q = q + (-worst).clamp_min(0)[..., None]*n
    distance, normals = geometry(q, nominal, cfg)
    for i in range(normals.shape[-2]):
        n = normals[..., i, :]
        relative = velocity-nominal_velocity if i == 6 and nominal_velocity is not None else velocity
        outward = (relative*n).sum(-1).clamp_max(0)
        velocity = velocity-torch.where(distance[..., i] < 1e-4, outward, 0)[..., None]*n
    # Explicitly flag rare incompatible constraints; never call this certified IK.
    invalid = (distance < -1e-4).any(-1)
    return q, velocity, invalid
