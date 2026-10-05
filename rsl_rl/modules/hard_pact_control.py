"""Simulator-neutral HardPACT PD/feedforward conversion, in physical Nm.

Callers choose raw or execution-clipped actions before forming q_des/tau_ff.
Actuator effects occur here exactly once. Neither current backend rate-limits
its non-QP controller; QP/held execution retain their separate hard rate box.
"""
import torch


def command_pair_gains(parameters):
    """Physical gains for already action-scaled q_des [rad], tau_ff [Nm]."""
    motor = parameters['control_motor_strength'].detach()
    feedback = parameters['control_feedback_weight'].detach()
    return (motor * feedback * parameters['control_kp'].detach(),
            motor * feedback * parameters['control_kd'].detach(),
            motor * parameters['control_feedforward_weight'].detach())


def command_pair_inputs(desired_position, feedforward, position, velocity, parameters):
    """Compact QP inputs rebuilt identically in rollout and differentiable replay."""
    kp,kd,ff = command_pair_gains(parameters)
    return dict(command_nominal=requested_torque_components(
        desired_position,feedforward,position,velocity,parameters)[0],
        command_kp=kp,command_kd=kd,
        command_ff_gain=ff,command_desired_position=desired_position,
        command_feedforward=feedforward,
        command_enabled=((kp.abs()>1e-12)|(ff.abs()>1e-12)))


def allocate_command_correction(u, parameters, position_share):
    """K delta_q = rho*u, F delta_ff = (1-rho)*u; no action re-scaling.

    A disabled branch transfers its share to the other branch. If both gains
    vanish no correction is realizable (the QP fixes that coordinate to zero).
    Safe denominators prevent NaNs/invalid VJPs at endpoint shares/zero gains.
    """
    kp, _, ff = command_pair_gains(parameters)
    pos, feed = kp.abs() > 1e-12, ff.abs() > 1e-12
    rho = torch.where(pos, torch.where(feed, u.new_tensor(position_share), 1.), 0.)
    delta_q = torch.where(pos, rho*u/torch.where(pos,kp,1.), 0.)
    delta_ff = torch.where(feed, (1-rho)*u/torch.where(feed,ff,1.), 0.)
    return delta_q, delta_ff, rho


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
