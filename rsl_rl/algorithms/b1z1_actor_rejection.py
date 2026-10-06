"""Actor-only force counterfactuals; fixed mechanics and frozen learned physics."""
import torch
import torch.nn.functional as F
from .b1z1_bard_pinn import yaw_world


def term_weight(a, name):
    if not a.cfg.get("actor_phys_" + name + "_rejection_enabled", False):
        return 0.
    return getattr(a, "rejection_progress", 0.) * a.cfg.get("actor_phys_" + name + "_rejection_weight", 0.)


def grf_blend(a):
    # Actor physics uses frozen decoder predictions without a reliability curriculum.
    return float(a.cfg.get("actor_phys_live_grf_enabled", False))


def bounded_torque(a, actions, state, limit):
    """Same PD/FF/motor-strength mapping and torque saturation as actor rollout."""
    actions = actions.clamp(-a.cfg["clip_actions"], a.cfg["clip_actions"])
    return a._coupled_torque(actions, state.detach()).clamp(-limit.detach(), limit.detach())


def decoder_grf(a, context, torque, stored, quaternion):
    radius = a.cfg.get("actor_phys_grf_torque_trust_radius", 0.)
    clipped = torque.new_zeros(())
    if radius > 0:
        delta = torque - stored.detach()
        clipped = (delta.abs() > radius).to(torque.dtype).mean()
        torque = stored.detach() + delta.clamp(-radius, radius)
    prediction = a.actor_critic.physics_decoder.predict_grf_actor(
        context["z"], context["explicit_condition"], torque)
    # Match normal GRF units and yaw-to-world conversion; no new output transform.
    world = yaw_world(prediction / a.cfg["grf_scale"], quaternion.detach())
    return world.reshape(len(torque), -1, 3), clipped


def neutral_torque(a, obs, context, state, limit, *, return_actions=False):
    model = a.actor_critic
    saved = {name: getattr(model, name) for name in (
        "last_tracking_error_sq", "last_film_magnitude", "last_film_identity_deviation")}
    neutral = {name: value.detach() for name, value in context.items()}
    neutral["ee_force"] = torch.zeros_like(neutral["ee_force"])
    try:
        with torch.no_grad():
            position, feedforward = model.actor_forward(obs.detach(), neutral, neutral)
            actions = torch.cat((position, feedforward), -1) if model.action_mode == "coupled" else position
            result = (bounded_torque(a, actions, state, limit).detach(), neutral)
            return (*result, actions.detach()) if return_actions else result
    finally:
        for name, value in saved.items():
            setattr(model, name, value)


def project_arm_force(mass, jacobian, delta, damping, *, return_projection=False):
    """Solve Ma[y,X]=[delta,J^T] once; then (JX+eps I)F=Jy."""
    mass, jacobian = mass.detach(), jacobian.detach()
    factor, info = torch.linalg.cholesky_ex(mass)
    valid = (info == 0) & torch.isfinite(mass).flatten(1).all(-1)
    result = delta.new_zeros(len(delta), jacobian.shape[1])
    condition = delta.new_zeros(len(delta))
    projection = delta.new_zeros(len(delta), jacobian.shape[1], jacobian.shape[2]) if return_projection else None
    if valid.any():
        rows = valid.nonzero().flatten()
        j = jacobian[rows]
        rhs = torch.cat((delta[rows, :, None], j.transpose(-1, -2)), -1)
        solution = torch.cholesky_solve(rhs, factor[rows])
        matrix = j @ solution[:, :, 1:] + damping * torch.eye(j.shape[1], device=j.device, dtype=j.dtype)
        right = j @ solution[:, :, :1]
        if return_projection:
            # X^T = J M^-1: append A's RHS to the existing solve, never refactor or solve again.
            right = torch.cat((right, solution[:, :, 1:].transpose(-1, -2).detach()), -1)
        solved, status = torch.linalg.solve_ex(matrix, right)
        force = solved[..., 0]
        good = (status == 0) & torch.isfinite(force).all(-1)
        result = result.index_copy(0, rows[good], force[good])
        if return_projection:
            projection[rows[good]] = solved[good, :, 1:].detach()
        with torch.no_grad():
            condition[rows[good]] = torch.linalg.cond(matrix[good])
        valid[rows[~good]] = False
    result = (result, valid, condition)
    return (*result, projection.detach()) if return_projection else result


def allocation_weight(a):
    """Inner coefficient: the outer actor schedule supplies coef*r, leaving weight*r*p^2."""
    from .b1z1_actor_physics import enabled
    if (not enabled(a.cfg) or a.actor_critic.action_mode != "coupled"
            or term_weight(a, "arm") <= 0):
        return 0.
    return (a.cfg.get("actor_phys_force_allocation_weight", 0.)
            * getattr(a, "rejection_progress", 0.)**2 / a.cfg["actor_phys_coef"])


def force_allocation(a, actions, neutral_actions, state, ids, projection, gate, generated):
    """Only the conditioned position command is live; nominal stabilization cancels."""
    count = a.actor_critic.num_actions
    actions = actions.clamp(-a.cfg["clip_actions"], a.cfg["clip_actions"])
    neutral_actions = neutral_actions.detach().clamp(-a.cfg["clip_actions"], a.cfg["clip_actions"])
    state, projection = state.detach(), projection.detach()
    pd = a._coupled_torque(actions[:, :count], state)
    with torch.no_grad():
        neutral_pd = a._coupled_torque(neutral_actions[:, :count], state)
    delta_pd = (pd-neutral_pd).index_select(1, ids)
    f_pd = (projection @ delta_pd.unsqueeze(-1)).squeeze(-1)
    raw, _ = force_residual(f_pd, torch.zeros_like(f_pd), gate,
        a.cfg["actor_phys_arm_force_scale"], a.cfg["actor_phys_arm_force_axis_weights"],
        a.cfg["actor_phys_huber_delta"])
    with torch.no_grad():
        delta_total = (a._coupled_torque(actions.detach(), state)
                       - a._coupled_torque(neutral_actions, state)).index_select(1, ids)
        f_ff = (projection @ (delta_total-delta_pd.detach()).unsqueeze(-1)).squeeze(-1)
        # Clipping is nonlinear: keep its residual explicit rather than calling it FF.
        discrepancy = generated.detach()-f_pd.detach()-f_ff
        pd_norm, ff_norm = f_pd.detach().norm(dim=-1), f_ff.norm(dim=-1)
        metrics = dict(allocation_raw=raw.detach(), allocation_pd_force_norm=pd_norm.mean(),
            allocation_ff_force_norm=ff_norm.mean(),
            allocation_total_force_norm=generated.detach().norm(dim=-1).mean(),
            allocation_ff_fraction=(ff_norm/(ff_norm+pd_norm+1e-8)).mean(),
            allocation_decomposition_error=discrepancy.norm(dim=-1).mean())
    return raw, metrics


def force_residual(generated, external, gate, scale, weights, delta):
    # External forces act ON the robot; generated compensation must oppose them.
    residual = generated + external.detach()
    error = residual * residual.new_tensor(weights) / scale
    per_sample = F.huber_loss(error, torch.zeros_like(error), reduction="none", delta=delta).mean(-1)
    return (per_sample * gate.detach()).mean(), residual


def prepare(a, batch, actions, context, data, fixed, torque, detached_grf, ee, base_force):
    """One shared neutral pass for both new losses; live GRFs also serve rollout."""
    zero = torque.sum() * 0.
    arm_weight, base_weight = term_weight(a, "arm"), term_weight(a, "base")
    allocation = allocation_weight(a)
    blend = grf_blend(a)
    metrics = {name: zero.detach() for name in (
        "arm_raw", "arm_weighted", "base_raw", "base_weighted", "external_force_norm",
        "generated_arm_force_norm", "arm_residual", "live_stance_force_norm",
        "neutral_stance_force_norm", "incremental_stance_force_norm", "base_residual",
        "arm_direction_cosine", "base_direction_cosine", "projection_condition",
        "projection_failure_fraction", "trust_clipped_fraction")}
    metrics.update(progress=torque.new_tensor(getattr(a, "rejection_progress", 0.)),
                   arm_weight=torque.new_tensor(arm_weight), base_weight=torque.new_tensor(base_weight),
                   grf_blend=torque.new_tensor(blend))
    live_grf = detached_grf
    if blend > 0:
        candidate, clipped = decoder_grf(a, context, torque, batch["nominal_torque"], data["state"][:, 3:7])
        finite = torch.isfinite(candidate).flatten(1).all(-1)
        candidate = torch.where(finite[:, None, None], candidate, detached_grf)
        live_grf = detached_grf + blend * (candidate - detached_grf)
        metrics["trust_clipped_fraction"] = clipped.detach()
    else:
        finite = torch.ones(len(torque), dtype=torch.bool, device=torque.device)
    if arm_weight <= 0 and (base_weight <= 0 or blend <= 0):
        return live_grf, zero, metrics
    # The forced branch is the already-computed deterministic actor mean supplied
    # by PPO. Only EE-force conditioning changes in this detached neutral pass.
    neutral_result = neutral_torque(a, batch["observations"], context, data["state"], data["torque_max"],
                                    **({"return_actions": True} if allocation > 0 else {}))
    neutral, neutral_context = neutral_result[:2]
    threshold = a.cfg["actor_phys_rejection_active_force"]
    active = (ee.norm(dim=-1) > threshold) | (base_force.norm(dim=-1) > threshold)
    active &= finite
    metrics["external_force_norm"] = (ee + base_force).norm(dim=-1).mean().detach()
    delta = a.cfg["actor_phys_huber_delta"]
    total = zero
    if arm_weight > 0:
        ids = torch.as_tensor(a.cfg["actor_phys_arm_dof_indices"], device=torque.device)
        base_width = fixed.mass_matrix.shape[-1] - torque.shape[-1]
        generalized_ids = ids + base_width
        mass = fixed.mass_matrix.index_select(1, generalized_ids).index_select(2, generalized_ids)
        jac = fixed.ee_jacobian[:, :3].index_select(2, generalized_ids)
        projected = project_arm_force(
            mass, jac, (torque-neutral).index_select(1, ids), a.cfg["actor_phys_rejection_damping"],
            **({"return_projection": True} if allocation > 0 else {}))
        generated, valid, condition = projected[:3]
        raw, residual = force_residual(generated, ee, active & valid,
            a.cfg["actor_phys_arm_force_scale"], a.cfg["actor_phys_arm_force_axis_weights"], delta)
        total = total + arm_weight * raw
        metrics.update(arm_raw=raw.detach(), arm_weighted=(arm_weight*raw).detach(),
            generated_arm_force_norm=generated.norm(dim=-1).mean().detach(),
            arm_residual=residual.norm(dim=-1).mean().detach(),
            arm_direction_cosine=F.cosine_similarity(generated, -ee, dim=-1).mean().detach(),
            projection_condition=condition.mean().detach(),
            projection_failure_fraction=(~valid).to(torque.dtype).mean())
        if allocation > 0:
            from .b1z1_actor_physics import scheduled_coefficient
            allocation_raw, allocation_metrics = force_allocation(
                a, actions, neutral_result[2], data["state"], ids, projected[3], active & valid, generated)
            total = total + allocation * allocation_raw
            effective = allocation * scheduled_coefficient(a)
            metrics.update(allocation_metrics,
                           allocation_weighted=effective * allocation_raw.detach(),
                           allocation_effective_weight=torque.new_tensor(effective))
    if base_weight > 0 and blend > 0:
        with torch.no_grad():
            neutral_grf, _ = decoder_grf(a, neutral_context, neutral,
                batch["nominal_torque"], data["state"][:, 3:7])
            neutral_grf = detached_grf + blend * (neutral_grf-detached_grf)
            finite_neutral = torch.isfinite(neutral_grf).flatten(1).all(-1)
            neutral_grf = torch.where(finite_neutral[:, None, None], neutral_grf, detached_grf)
        stance = data["stance"].detach().bool()
        live_stance = (live_grf * stance[:, :, None]).sum(1)[:, :2]
        neutral_stance = (neutral_grf * stance[:, :, None]).sum(1)[:, :2]
        increment = live_stance - neutral_stance
        gate = active & finite_neutral & (stance.sum(-1) >= a.cfg["actor_phys_rejection_min_contacts"])
        external = (ee + base_force)[:, :2]
        raw, residual = force_residual(increment, external, gate,
            a.cfg["actor_phys_base_force_scale"], a.cfg["actor_phys_base_force_axis_weights"], delta)
        total = total + base_weight * raw
        metrics.update(base_raw=raw.detach(), base_weighted=(base_weight*raw).detach(),
            live_stance_force_norm=live_stance.norm(dim=-1).mean().detach(),
            neutral_stance_force_norm=neutral_stance.norm(dim=-1).mean(),
            incremental_stance_force_norm=increment.norm(dim=-1).mean().detach(),
            base_residual=residual.norm(dim=-1).mean().detach(),
            base_direction_cosine=F.cosine_similarity(increment, -external, dim=-1).mean().detach())
    return live_grf, total, metrics
