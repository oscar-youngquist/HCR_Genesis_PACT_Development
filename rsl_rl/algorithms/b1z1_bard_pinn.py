"""Fixed-mechanics B1Z1 PINNs using the HardPACT reductions and gradient contract."""

from types import SimpleNamespace
from contextlib import contextmanager
import warnings
import torch
from rsl_rl.algorithms.pc_grad import PCGrad
from legged_gym.utils.math_utils import quat_apply
from rsl_rl.algorithms.hard_pact_bard import (
    corrected_bard_inverse_dynamics_loss, differentiable_bard_rollout_loss,
    measured_contact_generalized_force,
)


def yaw_world(value, quaternion):
    x, y, z, w = quaternion.unbind(-1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    q = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw),
                     torch.sin(yaw/2), torch.cos(yaw/2)), -1)
    shape = value.shape
    vectors = value.reshape(-1, shape[-1] // 3, 3)
    return quat_apply(q[:, None].expand(-1, vectors.shape[1], -1).reshape(-1, 4),
                      vectors.reshape(-1, 3)).reshape(shape)


@torch.no_grad()
def mechanics(backend, batch):
    """Cache measured q_t mechanics once per minibatch, outside both VJPs."""
    initial, state = batch["rollout_initial_state"], batch["dynamics_state"]
    zeros = initial.new_zeros
    terms = backend.evaluate(initial[:, :3], initial[:, 3:7], initial[:, 7:26],
        initial[:, 26:29], initial[:, 29:32], initial[:, 32:51],
        zeros((len(initial), 4, 3)), zeros((len(initial), 3)), zeros((len(initial), 6)),
        state[:, 175:176], state[:, 176:179], state[:, 179:180])
    result = SimpleNamespace(**{name: getattr(terms, name).detach().clone() for name in
        ("mass_matrix", "bias", "foot_jacobians", "ee_jacobian", "base_jacobian")})
    result.foot_jacobians = result.foot_jacobians[:, :, :3]
    return result


@torch.no_grad()
def physical_metrics(kind, loss, residual, metrics):
    """Keep force/moment and linear/angular velocity units separate."""
    metrics[f"{kind}/loss_raw"] = loss.detach()
    units = ("N", "Nm", "Nm", "Nm") if kind == "inverse" else ("mps", "radps", "radps", "radps")
    blocks = (("base_linear", slice(0, 3)), ("base_angular", slice(3, 6)),
              ("legs", slice(6, 18)), ("arm_gripper", slice(18, 25)))
    for (name, block), unit in zip(blocks, units):
        metrics[f"{kind}/{name}_mae_{unit}"] = residual[:, block].abs().mean()


def losses(model, context, batch, fixed, cfg, metrics=None):
    """Only z/force-head graphs survive; torque, states, labels and mechanics detach."""
    rows = ~(batch["dones"].bool() | batch["physics_invalid"].bool()).flatten()
    if not rows.any():
        zero = context["base_wrench"].sum() * 0 + model.predict_grf(context, batch["nominal_torque"]).sum() * 0
        return zero, zero
    context = {k: v[rows] for k, v in context.items()}
    batch = {k: v[rows] for k, v in batch.items()}
    fixed = SimpleNamespace(**{k: v[rows] for k, v in vars(fixed).items()})
    state, initial = batch["dynamics_state"], batch["rollout_initial_state"]
    grf = model.predict_grf(context, batch["nominal_torque"])
    # GRF supervision is in successor yaw; current disturbance supervision uses pre yaw.
    grf_world = yaw_world(grf / cfg["grf_scale"], state[:, 3:7]).reshape(-1, 4, 3)
    wrench = yaw_world(context["base_wrench"] / grf.new_tensor(cfg["base_wrench_scale"]), initial[:, 3:7])
    ee = yaw_world(context["ee_force"] / cfg["ee_force_scale"], initial[:, 3:7])
    ee_generalized = torch.einsum("bkn,bk->bn", fixed.ee_jacobian[:, :3], ee)
    mass_wrench = batch["mass_wrench"].detach()
    # As in HardPACT, shift the applied wrench from randomized torso CoM to
    # the base Jacobian's origin only after removing the label-only load.
    applied = wrench - mass_wrench
    lever = quat_apply(initial[:, 3:7].detach(), state[:, 176:179].detach())
    applied = torch.cat((applied[:, :3], applied[:, 3:] +
                         torch.cross(lever, applied[:, :3], dim=-1)), -1)
    measured_contact = measured_contact_generalized_force(fixed.foot_jacobians, state[:, 76:88])
    torque = batch["interval_torque"].detach()
    pre_v, post_v = initial[:, 26:51].detach(), state[:, 26:51].detach()
    dt = pre_v.new_full((len(pre_v), 1), cfg["dt"])
    acceleration = (post_v - pre_v) / dt
    required = torch.bmm(fixed.mass_matrix, acceleration.unsqueeze(-1)).squeeze(-1) + fixed.bias
    invalid = batch["dones"].bool().reshape(-1)
    masks = dict(push_event_mask=torch.zeros_like(invalid), reset_mask=invalid,
                 timeout_mask=torch.zeros_like(invalid), teleport_mask=batch["physics_invalid"].bool().reshape(-1))
    inverse = corrected_bard_inverse_dynamics_loss(
        required_generalized_force=required, foot_jacobians=fixed.foot_jacobians,
        base_jacobian=fixed.base_jacobian, interval_executed_torque=torque,
        interval_grf_world=grf_world, total_wrench_world=applied + mass_wrench,
        mass_com_wrench_world=mass_wrench, measured_generalized_contact_force=measured_contact,
        additional_generalized_force=ee_generalized, **masks)
    if metrics is not None:
        physical_metrics("inverse", inverse.loss, inverse.residual, metrics)
        metrics["valid_samples"] = len(state)
    if not cfg.get("use_pinn_rollout_loss", True):
        return inverse.loss, inverse.loss * 0
    # Detached M makes torch.linalg.solve's backward an RHS-only M^-T solve.
    rollout_context = SimpleNamespace(foot_jacobians=fixed.foot_jacobians,
        base_jacobian=fixed.base_jacobian, pre_v_canonical=pre_v, post_v_canonical=post_v,
        forward_dynamics=lambda force: torch.linalg.solve(
            fixed.mass_matrix, (force - fixed.bias).unsqueeze(-1)).squeeze(-1))
    rollout = differentiable_bard_rollout_loss(context=rollout_context,
        control_torque=torque, interval_grf_world=grf_world,
        applied_wrench_world=applied, control_dt=dt,
        additional_generalized_force=ee_generalized, **masks)
    if metrics is not None:
        physical_metrics("rollout", rollout.loss,
                         rollout.predicted_velocity - rollout.target_velocity, metrics)
    return inverse.loss, rollout.loss


def configure_optimizers(algorithm, actor_groups, auxiliary_groups):
    """Disjoint actor/critic, history encoder, and all decoder ownership."""
    a = algorithm
    a.decoder_parameters = list(a.privileged_decoder.parameters()) + list(a.actor_critic.physics_decoder.parameters()) + list(a.actor_critic.explicit_decoder.parameters())
    decoder_ids = {id(p) for p in a.decoder_parameters}
    def groups(decoders):
        return [{**g, "params": selected} for g in auxiliary_groups
                if (selected := [p for p in g["params"] if (id(p) in decoder_ids) == decoders])]
    a.actor_optimizer = PCGrad(torch.optim.AdamW(actor_groups, lr=a.learning_rate), reduction="sum", owned_only=True)
    a.ppo_parameters = [p for g in actor_groups for p in g["params"]]
    lr = a.cfg.get("adaptation_learning_rate", 1e-5)
    a.auxiliary_optimizer = torch.optim.AdamW(groups(False), lr=lr)
    a.decoder_optimizer = torch.optim.AdamW(groups(True), lr=lr)
    a.enc_parameters = [p for g in a.auxiliary_optimizer.param_groups for p in g["params"]]
    a.encoder_pcgrad = PCGrad(a.auxiliary_optimizer, reduction="sum", owned_only=True)
    a.decoder_pcgrad = PCGrad(a.decoder_optimizer, reduction="sum", owned_only=True)
    owners = [set(map(id, p)) for p in (a.ppo_parameters, a.enc_parameters, a.decoder_parameters)]
    if any(owners[i] & owners[j] for i in range(3) for j in range(i)):
        raise RuntimeError("B1Z1 optimizer parameter ownership overlaps")
    expected = {id(p) for m in (a.actor_critic, a.privileged_decoder) for p in m.parameters() if p.requires_grad}
    if set.union(*owners) != expected:
        raise RuntimeError("B1Z1 optimizer partition omits trainable parameters")


def restore_optimizers(algorithm, checkpoint):
    """Old overlapping/encoder-plus-explicit moments cannot map by group order."""
    if checkpoint.get("optimizer_partition_version") != 2:
        warnings.warn("Legacy B1Z1 optimizer partition: restoring weights with fresh optimizer states.")
        return
    algorithm.actor_optimizer.optimizer.load_state_dict(checkpoint["actor_optimizer"])
    algorithm.auxiliary_optimizer.load_state_dict(checkpoint["auxiliary_optimizer"])
    algorithm.decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])


@contextmanager
def frozen(parameters):
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def combined_pinn_loss(inverse, rollout, cfg):
    """Component weights inside the outer scheduled pinn_loss_weight."""
    return (cfg.get("pinn_inverse_weight", 1.0) * inverse
            + cfg.get("pinn_rollout_weight", 1.0) * rollout)


def auxiliary_backward(algorithm, optimizer, supervised, physics):
    """Use the configured sign for projection, never for the loss magnitude."""
    weight = getattr(algorithm, "pinn_weight", 0.0)
    if weight > 0 and physics.requires_grad:
        project = (optimizer.pc_backward_ppgrad if algorithm.cfg["pinn_loss_weight"] < 0
                   else optimizer.pc_backward_pinn)
        project([supervised, weight * physics])
    else:
        optimizer.pc_backward([supervised])


def auxiliary_step(a, batch, valid, iteration):
    """Two disjoint PCGrad phases using one stochastic sample and one mechanics cache."""
    rows = valid.flatten().bool()
    arguments = dict(obs_hist_batch=batch["histories"], obs_target=batch["next_privileged"],
        labels=batch["explicit_targets"], valid=valid, iteration=iteration,
        nominal_torque=batch["nominal_torque"], update=False)
    zero = batch["histories"].new_zeros(())
    a.bard_phase_metrics = {}
    if not rows.any():
        return a._compute_vae_loss(**arguments), zero, zero
    selected = {name: value[rows] for name, value in batch.items() if name != "indices"}
    fixed = None
    physics_active = getattr(a, "pinn_weight", 0.0) > 0
    if physics_active and a.bard_auxiliary:
        fixed = SimpleNamespace(**{name: value[batch["indices"][rows]]
            for name, value in vars(a.bard_mechanics_cache).items()})
    def phase_losses(context, phase):
        if not physics_active:
            return zero, zero
        if a.bard_auxiliary:
            measured = {}
            result = losses(a.actor_critic, context, selected, fixed, a.cfg, metrics=measured)
            count = measured.pop("valid_samples", 0)
            # Loss scaling, not physical-unit conversion; sign only selects PCGrad.
            for component in ("inverse", "rollout"):
                raw = measured.get(f"{component}/loss_raw")
                if raw is not None:
                    weighted = raw * a.cfg.get(f"pinn_{component}_weight", 1.0)
                    measured[f"{component}/loss_unscaled"] = raw
                    measured[f"{component}/loss_component_weighted"] = weighted
                    measured[f"{component}/loss_scaled"] = a.pinn_weight * weighted
            # Weight by valid transitions, not minibatch size or reset frequency.
            for name, value in measured.items():
                for prefix in ("PINN", f"PINN/{phase}"):
                    key = f"{prefix}/{name}"
                    total, samples = a.pinn_metric_sums.get(key, (value.new_zeros(()), 0))
                    a.pinn_metric_sums[key] = (total + value * count, samples + count)
            return result
        # Pinocchio retains its force gate/formulation, but cannot update the actor.
        with torch.no_grad():
            actions, source_valid = a._physics_actions(selected)
        context = {**context, "base_quat_t": selected["rollout_initial_state"][:, 3:7],
                   "rollout_initial_state": selected["rollout_initial_state"]}
        prediction = torch.cat((a.actor_critic.predict_grf(context, selected["nominal_torque"]),
                                context["base_wrench"], context["ee_force"]), -1)
        inverse = a._pinn_loss(actions, context, prediction, selected["dynamics_state"], source_valid)
        rollout = zero
        if a.cfg["use_pinn_rollout_loss"]:
            rollout, _ = a._rollout_pinn_loss(actions, prediction, selected["dynamics_state"],
                                             selected["rollout_initial_state"], source_valid)
        return inverse, rollout
    phase_values = []
    a.encoder_pcgrad.zero_grad()
    a.decoder_pcgrad.zero_grad()
    with frozen(a.decoder_parameters):
        aux = a._compute_vae_loss(**arguments)
        encoder_context = aux["context"]
        inv, roll = phase_losses(encoder_context, "encoder")
        physics = combined_pinn_loss(inv, roll, a.cfg)
        auxiliary_backward(a, a.encoder_pcgrad, aux["loss"], physics)
        phase_values.append((inv.detach(), roll.detach()))
    detached = {name: value.detach() for name, value in encoder_context.items()}
    dec = a._compute_vae_loss(**arguments, context_override=detached)
    inv, roll = phase_losses(dec["context"], "decoder")
    physics = combined_pinn_loss(inv, roll, a.cfg)
    auxiliary_backward(a, a.decoder_pcgrad, dec["loss"], physics)
    phase_values.append((inv.detach(), roll.detach()))
    # Complete both VJPs before changing parameters used by the shared sample.
    for optimizer, parameters in ((a.auxiliary_optimizer, a.enc_parameters),
                                  (a.decoder_optimizer, a.decoder_parameters)):
        torch.nn.utils.clip_grad_norm_(parameters, a.max_grad_norm)
        optimizer.step()
    a.bard_phase_metrics = {f"pinn_{phase}_{kind}": values[i]
        for phase, values in zip(("encoder", "decoder"), phase_values)
        for i, kind in enumerate(("inverse", "rollout"))}
    return aux, sum(v[0] for v in phase_values)/2, sum(v[1] for v in phase_values)/2


@torch.no_grad()
def cache_rollout(a):
    """Evaluate bounded BARD batches once; shuffled epochs only index detached terms."""
    storage = a.storage
    flat = {name: getattr(storage, name).flatten(0, 1) for name in
            ("rollout_initial_state", "dynamics_state")}
    invalid = (storage.dones | storage.physics_invalid.bool()).flatten()
    pieces = {}
    capacity = a.dynamics_backend.batch_capacity
    for start in range(0, len(flat["dynamics_state"]), capacity):
        chunk = {k: v[start:start+capacity].clone() for k, v in flat.items()}
        # Reset/teleport rows are excluded by losses; do not feed their state to BARD.
        bad = invalid[start:start+capacity]
        for value in chunk.values():
            value[bad] = 0
        chunk["rollout_initial_state"][bad, 6] = 1
        terms = mechanics(a.dynamics_backend, chunk)
        for name, value in vars(terms).items():
            pieces.setdefault(name, []).append(value)
    return SimpleNamespace(**{name: torch.cat(values) for name, values in pieces.items()})
