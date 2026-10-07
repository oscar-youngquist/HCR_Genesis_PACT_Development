"""Simulator-neutral HardPACT PD/feedforward conversion, in physical Nm.

Callers choose raw or execution-clipped actions before forming q_des/tau_ff.
Actuator effects occur here exactly once. Neither current backend rate-limits
its non-QP controller; QP/held execution retain their separate hard rate box.
"""
import torch


def feedforward_for_total_torque(total, desired, position, velocity, parameters, nominal_ff):
    """Keep q_des untouched. Convert selected physical total Nm exactly once.

    A zero feedforward gain cannot realize an arbitrary correction: preserve
    that command and report reconstruction availability. The QP fixes its u
    to zero; impossible motor constraints then use the existing fallback.
    """
    kp,kd,gain=command_pair_gains(parameters)
    pd=kp*(desired-position.detach())-kd*velocity.detach()
    enabled=gain.abs()>1e-12
    command=torch.where(enabled,(total-pd)/torch.where(enabled,gain,1.),nominal_ff)
    return command, enabled


def pd_prediction_maps(acceleration_map, acceleration_offset, origin, kp, kd, q, v, horizon, beta):
    """Frozen full 18-DoF dynamics, x=[u12,f12], two internal H/2 steps.

    At l: tau_l=origin+u-K(q_l-q0)-D(v_l-v0),
    a_l=a0(x)+M^-1 S^T(tau_l-(origin+u)). Integrate each NEW acceleration:
    q+=h*v+beta*h²*a_j; v+=h*a_j. beta=1: semi-implicit Euler;
    beta=.5: local constant-acceleration step. These internal points do not
    alter simulator time or QP dispatch frequency. Frozen full mechanics
    retain base/joint/contact coupling; no quaternion relinearization occurs.
    Returns maps/offsets ordered [q_mid,q_end,v_mid,v_end,tau0,tau_mid,tau_end].
    """
    batch=origin.shape[0];h=horizon.reshape(-1,1)/2
    eye=torch.eye(24,device=origin.device,dtype=origin.dtype)[:12].expand(batch,-1,-1)
    qm=torch.zeros_like(eye);vm=torch.zeros_like(eye);qo=q.detach();vo=v.detach()
    tm=eye;to=origin
    qs=[];vs=[];ts=[(tm,to)]
    B=acceleration_map[:,:,:12]
    for _ in range(2):
        am=acceleration_map+B@(tm-eye)
        ao=acceleration_offset+(B@(to-origin)[...,None]).squeeze(-1)
        qm=qm+h[:,:,None]*vm+beta*h.square()[:,:,None]*am[:,6:]
        qo=qo+h*vo+beta*h.square()*ao[:,6:]
        vm=vm+h[:,:,None]*am[:,6:];vo=vo+h*ao[:,6:]
        qs.append((qm,qo));vs.append((vm,vo))
        tm=eye-kp[:,:,None]*qm-kd[:,:,None]*vm
        to=origin-kp*(qo-q.detach())-kd*(vo-v.detach())
        ts.append((tm,to))
    return torch.stack([m for m,b in qs+vs+ts],1),torch.stack([b for m,b in qs+vs+ts],1)


def command_pair_gains(parameters):
    """Physical gains for already action-scaled q_des [rad], tau_ff [Nm]."""
    motor = parameters['control_motor_strength'].detach()
    feedback = parameters['control_feedback_weight'].detach()
    return (motor * feedback * parameters['control_kp'].detach(),
            motor * feedback * parameters['control_kd'].detach(),
            motor * parameters['control_feedforward_weight'].detach())


def command_pair_inputs(desired_position, feedforward, position, velocity, parameters,
                        position_lower=None, position_upper=None, feedforward_limits=None):
    """Compact QP inputs rebuilt identically in rollout and differentiable replay."""
    kp,kd,ff = command_pair_gains(parameters)
    result = dict(command_nominal=requested_torque_components(
        desired_position,feedforward,position,velocity,parameters)[0],
        command_kp=kp,command_kd=kd,
        command_ff_gain=ff,command_desired_position=desired_position,
        command_feedforward=feedforward,
        command_enabled=((kp.abs()>1e-12)|(ff.abs()>1e-12)))
    if position_lower is not None:
        a0=kp*(position_lower-desired_position);a1=kp*(position_upper-desired_position)
        amin,amax=torch.minimum(a0,a1),torch.maximum(a0,a1)
        if feedforward_limits is None:
            bmin=torch.full_like(ff,-torch.inf);bmax=-bmin
        else:
            limit=feedforward.new_tensor(feedforward_limits)
            b0=ff*(-limit-feedforward);b1=ff*(limit-feedforward)
            bmin,bmax=torch.minimum(b0,b1),torch.maximum(b0,b1)
        bmin=torch.where(ff.abs()>1e-12,bmin,0.)
        bmax=torch.where(ff.abs()>1e-12,bmax,0.)
        result.update(allocation_a_min=amin,allocation_a_max=amax,
                      allocation_b_min=bmin,allocation_b_max=bmax)
    return result


def allocate_command_correction(u, parameters, position_share, bounds=None):
    """K delta_q = rho*u, F delta_ff = (1-rho)*u; no action re-scaling.

    A disabled branch transfers its share to the other branch. If both gains
    vanish no correction is realizable (the QP fixes that coordinate to zero).
    Safe denominators prevent NaNs/invalid VJPs at endpoint shares/zero gains.
    """
    kp, _, ff = command_pair_gains(parameters)
    pos, feed = kp.abs() > 1e-12, ff.abs() > 1e-12
    rho = torch.where(pos, torch.where(feed, u.new_tensor(position_share), 1.), 0.)
    a=rho*u
    if bounds is not None:
        lower=torch.maximum(bounds['allocation_a_min'],u-bounds['allocation_b_max'])
        upper=torch.minimum(bounds['allocation_a_max'],u-bounds['allocation_b_min'])
        # Infeasibility is rejected by the QP bounds; keep rejected-row helper
        # arithmetic finite. Callers must mask failed candidates, never execute it.
        a=torch.maximum(torch.minimum(position_share*u,upper),lower)
        a=torch.where(lower<=upper,a,torch.zeros_like(a))
    delta_q = torch.where(pos, a/torch.where(pos,kp,1.), 0.)
    delta_ff = torch.where(feed, (u-a)/torch.where(feed,ff,1.), 0.)
    return delta_q, delta_ff, rho


def allocation_deviation_loss(u, a, torque_scale, preference, valid):
    """Full unblended accepted-row mean; endpoint preference disables penalty."""
    u,a=u[valid],a[valid]
    if not 0<preference<1 or not u.shape[0]:
        return (u.sum()+a.sum())*0
    return ((a-preference*u).square()/(preference*(1-preference)*torque_scale.square())).sum(-1).mean()


def command_pair_sample(num_envs, fraction, later_substep, device):
    """Fixed-size random subset; one shared later dispatch, one replay per row."""
    selected=torch.zeros(num_envs,device=device,dtype=torch.long)
    rows=torch.randperm(num_envs,device=device)[:round(num_envs*fraction)]
    selected[rows]=later_substep
    return selected


def held_command_model(dt, beta, kp, kd, velocity, substeps=4):
    """Constant interval-average a, with frozen M/J and held commands.

    At endpoint k: v_k=v0+k*dt*a; q_k=q0+k*dt*v0+c_k*a,
    c_k=dt²[k(k-1)/2+beta*k]. beta=1 is semi-implicit Euler;
    beta=.5 is constant-acceleration integration. At application k=0..D-1:
    tau_k=tau0+u-K*k*dt*v0-(K*c_k+Kd*k*dt)*a_joint.
    Averaging these application torques gives the implicit effective inertia
    M_eff=M+S^T diag(mean(K*c_k+Kd*k*dt)) S, not M at dt=control_dt.
    """
    k = torch.arange(substeps+1,device=dt.device,dtype=dt.dtype)[None,:,None]
    time = dt.reshape(-1,1,1)*k
    c = dt.reshape(-1,1,1).square()*(k*(k-1)/2+beta*k)
    drift = kp[:,None]*time[:,:-1]*velocity.detach()[:,None]
    decay = kp[:,None]*c[:,:-1]+kd[:,None]*time[:,:-1]
    return time, c, drift, decay


def requested_torque_components(desired_position, feedforward, position, velocity, parameters):
    """Return total, physical feedback/feedforward, and legacy unweighted PD.

    The desired position is absolute (default pose is already included).
    Measured controller parameters are detached; action gradients are retained.
    """
    p = {key: parameters[key].detach() for key in (
        "control_kp", "control_kd", "control_motor_strength",
        "control_feedback_weight", "control_feedforward_weight",
    )}
    pd = p["control_kp"] * (desired_position - position.detach()) - p["control_kd"] * velocity.detach()
    fb = p["control_feedback_weight"] * pd
    ff = p["control_feedforward_weight"] * feedforward
    motor = p["control_motor_strength"]
    return motor * (ff + fb), motor * fb, motor * ff, pd


def bounded_nominal_torque(desired_position, feedforward, position, velocity, parameters):
    """The non-QP actuator command: clipped actions -> requested Nm -> bounds.

    These are actuator bounds, not the QP rate box. The latter continues to
    reference the previous *final executed* command inside the existing QP.
    """
    requested = requested_torque_components(
        desired_position, feedforward, position, velocity, parameters,
    )[0]
    limits = parameters["control_torque_limits"].detach()
    return torch.clamp(requested, -limits, limits)
