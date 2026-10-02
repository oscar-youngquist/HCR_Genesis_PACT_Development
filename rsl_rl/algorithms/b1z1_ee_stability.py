"""Near-target task-space damping using the existing actor rollout successor."""
import torch


def rotation_log(rotation):
    """SO(3) log with a small-angle limit and a diagonal-axis branch near pi."""
    skew = torch.stack((rotation[:, 2, 1]-rotation[:, 1, 2],
                        rotation[:, 0, 2]-rotation[:, 2, 0],
                        rotation[:, 1, 0]-rotation[:, 0, 1]), -1) * .5
    sine = skew.norm(dim=-1, keepdim=True)
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)-1)*.5).clamp(-1, 1)
    angle = torch.atan2(sine, cosine)
    regular = skew * torch.where(sine > 1e-6, angle / sine.clamp_min(1e-6), torch.ones_like(sine))
    # At pi the skew vanishes; the largest diagonal gives a well-conditioned axis.
    index = rotation.diagonal(dim1=-2, dim2=-1).argmax(-1)
    symmetric = rotation + rotation.transpose(-1, -2) + 2*torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    axis = symmetric.gather(2, index[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    axis = torch.nn.functional.normalize(axis, dim=-1, eps=1e-6)
    axis = axis * torch.where((axis*skew).sum(-1, keepdim=True) < 0, -1., 1.)
    return torch.where(cosine < -.9999, angle*axis, regular)


def objective(backend, predicted, data, cfg):
    """All references/current-state terms are fixed; only successor FK/Jv is live."""
    zero = predicted.new_zeros(())
    names = ("raw", "weighted", "twist", "energy_violation", "gate", "active_fraction",
             "linear_speed", "angular_speed")
    metrics = {"ee_stability_"+name: zero for name in names}
    weight = cfg.get("actor_phys_ee_stability_weight", 0.)
    if weight == 0:
        return zero, metrics
    def setting(name):
        return cfg["actor_phys_ee_stability_"+name]
    orientation = cfg.get("actor_phys_ee_stability_use_orientation", False)
    def kinematics(state):
        return (backend.ee_pose_twist(state) if orientation else
                backend.ee_pose_twist(state, use_orientation=False))
    with torch.no_grad():
        pose, twist = kinematics(data["state"].detach())
        target = data["ee_target"].detach()
        current_target = data["ee_stability_current_target"].detach()
        desired_twist = torch.zeros_like(twist)
        desired_twist[:, :3] = (target-current_target) / cfg["dt"]
    next_pose, next_twist = kinematics(predicted)
    finite = torch.isfinite(next_pose).flatten(1).all(-1) & torch.isfinite(next_twist).all(-1)
    finite &= torch.isfinite(pose).flatten(1).all(-1) & torch.isfinite(twist).all(-1)
    if not finite.any():
        return zero, metrics
    pose, twist, next_pose, next_twist, target, current_target, desired_twist = [
        x[finite] for x in (pose, twist, next_pose, next_twist, target, current_target, desired_twist)]
    ep = (pose[:, :3, 3] if orientation else pose) - current_target
    en = (next_pose[:, :3, 3] if orientation else next_pose) - target
    # Reuse the task's yaw-relative default orientation, held at measured yaw.
    er, ern = torch.zeros_like(ep), torch.zeros_like(en)
    if orientation and "ee_stability_rotation" in data:
        rotation = data["ee_stability_rotation"].detach()[finite]
        er = rotation_log(pose[:, :3, :3] @ rotation.transpose(-1, -2))
        ern = rotation_log(next_pose[:, :3, :3] @ rotation.transpose(-1, -2))
    error, next_error = ((torch.cat((ep, er), -1), torch.cat((en, ern), -1))
                         if orientation else (ep, en))
    width = 6 if orientation else 3
    we = predicted.new_tensor(setting("pose_weights")[:width])
    wx = predicted.new_tensor(setting("twist_weights")[:width])
    relative, next_relative = twist-desired_twist, next_twist-desired_twist
    energy = (error.square()*we).sum(-1) + setting("beta")*(relative.square()*wx).sum(-1)
    next_energy = (next_error.square()*we).sum(-1) + setting("beta")*(next_relative.square()*wx).sum(-1)
    distance = (ep/setting("position_radius")).square().sum(-1)
    if orientation:
        distance = distance + (er/setting("rotation_radius")).square().sum(-1)
    gate = torch.exp(-.5*distance)
    gate *= desired_twist[:, :3].norm(dim=-1) <= setting("target_speed_threshold")
    gate = gate.detach()
    damping = (next_relative.square()*wx).sum(-1)
    violation = (next_energy-(1-setting("rho"))*energy.detach()-setting("energy_slack")).clamp_min(0)
    raw = (gate*(setting("twist_weight")*damping + setting("energy_weight")*violation.square())).mean()
    weighted = weight*raw
    values = (raw, weighted, (gate*damping).mean(), (gate*violation).mean(), gate.mean(),
              (gate > .01).to(predicted.dtype).mean(), next_twist[:, :3].norm(dim=-1).mean(),
              next_twist[:, 3:].norm(dim=-1).mean() if orientation else zero)
    metrics.update({"ee_stability_"+name: value.detach() for name, value in zip(names, values)})
    return weighted, metrics
