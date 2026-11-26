"""Evaluation-only capture and causal, detached inverse-dynamics baselines.

No imports from this module are used by training or control. Offline estimates
are timestamped at the END of their backward-difference interval. Nominal BARD
mechanics are from its START. Forces are LWA/world, moments about base-link origin.
"""
import torch


def inverse_rotate(q, v):
    t = 2 * torch.cross(q[..., :3], v, dim=-1)
    return v - q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


def rotate(q, v):
    return inverse_rotate(torch.cat((-q[..., :3], q[..., 3:]), -1), v)


def acceleration(q, before, after, dt):
    """World link-origin twists + canonical joint rates -> BARD body-twist qdd.

    Express interval derivative at the start orientation. Linear transport is
    -omega_body x v_body; angular self transport is zero. No CoM velocity is used.
    """
    diff = (after - before) / dt
    v = inverse_rotate(q, before[..., :3])
    w = inverse_rotate(q, before[..., 3:6])
    return torch.cat((inverse_rotate(q, diff[..., :3]) - torch.cross(w, v, dim=-1),
                      inverse_rotate(q, diff[..., 3:6]), diff[..., 6:]), -1)


@torch.no_grad()
def estimate(mass, bias, jac, base_jac, acc, torque, contact, damping=1e-4):
    """Full coupled 12x12 actuated-row damped least squares, then 6x6 base solve.

    Damping is lambda in A.T A + lambda I, units m² (joint force Jacobian).
    Hard contact gates zero columns; no friction/force clipping is performed.
    Invalid inputs are sanitized only for safe linalg and marked invalid outputs.
    """
    if not 0 < damping < float('inf'):
        raise ValueError('damping must be finite and positive')
    finite = torch.stack([x.flatten(1).isfinite().all(-1) for x in
        (mass, bias, jac, base_jac, acc, torque, contact)], -1).all(-1)
    mass,bias,jac,base_jac,acc,torque,contact = [torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.) for x in
        (mass,bias,jac,base_jac,acc,torque,contact)]
    residual = (mass @ acc[..., None]).squeeze(-1) + bias
    residual[:, 6:] -= torque
    jt = jac.flatten(1, 2).transpose(-1, -2)
    a = jt[:, 6:] * contact.repeat_interleave(3, -1)[:, None]
    gram = a.transpose(-1,-2) @ a
    eig = torch.linalg.eigvalsh(gram)
    system = gram + damping * torch.eye(12, device=gram.device, dtype=gram.dtype)
    f, info = torch.linalg.solve_ex(system, a.transpose(-1,-2) @ residual[:,6:,None], check_errors=False)
    f = f.squeeze(-1) * contact.repeat_interleave(3, -1)
    remaining = residual - (jt @ f[...,None]).squeeze(-1)
    b = base_jac.transpose(-1,-2)[:,:6]
    wrench, wi = torch.linalg.solve_ex(b, remaining[:,:6,None], check_errors=False)
    wrench = wrench.squeeze(-1)
    valid = finite & (info == 0) & (wi == 0) & f.isfinite().all(-1) & wrench.isfinite().all(-1)
    base_error = (b @ wrench[...,None]).squeeze(-1) - remaining[:,:6]
    return dict(force=f.reshape(-1,4,3).masked_fill(~valid[:,None,None],float('nan')),
        wrench=wrench.masked_fill(~valid[:,None],float('nan')), valid=valid,
        joint_residual=remaining[:,6:], base_residual=base_error,
        regularized_condition=(eig[:,-1]+damping)/(eig[:,0].clamp_min(0)+damping),
        rank=(eig > (eig[:,-1:]*1e-6).clamp_min(1e-12)).sum(-1),
        dynamics_residual=residual,input_finite=finite,force_solve_info=info,wrench_solve_info=wi)


def capture_pre(o):
    # All copies owned by record(); never retain mutable BARD workspace aliases.
    if o.t == 0 and o.k == 0:
        # One initialization snapshot, not a per-step host synchronization.
        import numpy as np
        np.savez_compressed(o.out/'shadow_realized_parameters.npz',
            **{k:v.detach().cpu().numpy() for k,v in o.e._canonical_randomized_parameters().items()})
    event = _start_timing(o)
    d=o.solve_data
    for key in ('mass_matrix','bias','foot_jacobians','base_jacobian','dt'):
        o.record('shadow_'+key,d[key])
    o.record('shadow_velocity_before',torch.cat((o.s._robot.data.root_link_vel_w,
        o.e._canonical_joint_state()[1]),-1))
    o.record('shadow_position_before',o.s._robot.data.root_link_pos_w)
    o.record('shadow_estimated_linear_body',o.actor.cenet_torso_velo[:,:3]/o.e.obs_scales.lin_vel)
    o.record('shadow_applied_wrench',o.s.hard_pact_wrench_at_base_origin_world(o.e._current_sustained_wrench_world))
    o.record('shadow_predicted_wrench',d['wrench_pred_world'])
    _end_timing(o,event)


def capture_post(o):
    event = _start_timing(o)
    o.record('shadow_velocity_after',torch.cat((o.s._robot.data.root_link_vel_w,
        o.e._canonical_joint_state()[1]),-1))
    o.record('shadow_position_after',o.s._robot.data.root_link_pos_w)
    o.record('shadow_force_truth',o.e.grf_processor.raw)
    o.record('shadow_force_label',o.e.grf_processor.clipped)
    o.record('shadow_applied_torque_readback',o.s._robot.data.applied_torque[:,o.s._dof_indices])
    # Ground-filtered all-body contact forces; normal internal contacts excluded.
    o.record('shadow_ground_contacts',o.s.ground_total_forces_world)
    _end_timing(o,event)


def _start_timing(o):
    if not 400 <= o.t*4+o.k < 656:return None
    event=torch.cuda.Event(enable_timing=True);event.record();return event


def _end_timing(o,event):
    if event is None:return
    end=torch.cuda.Event(enable_timing=True);end.record()
    if not hasattr(o,'shadow_timing_events'):o.shadow_timing_events=[]
    o.shadow_timing_events.append((event,end))
