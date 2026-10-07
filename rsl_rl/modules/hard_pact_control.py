"""Simulator-neutral HardPACT PD/feedforward conversion, in physical Nm.

Callers choose raw or execution-clipped actions before forming q_des/tau_ff.
Actuator effects occur here exactly once. Neither current backend rate-limits
its non-QP controller; QP/held execution retain their separate hard rate box.
"""
import torch


def effective_feedback_gains(parameters):
    """Physical Nm/rad and Nm/(rad/s), including actuator factors once."""
    gain = parameters['control_motor_strength'].detach() * parameters['control_feedback_weight'].detach()
    return dict(effective_kp=gain * parameters['control_kp'].detach(),
                effective_kd=gain * parameters['control_kd'].detach())


def execution_feedforward(selected, desired_position, position, velocity, parameters):
    """Reconstruct selected total Nm without changing the nominal position target.

    Zero-gain coordinates cannot represent an arbitrary torque via feedforward;
    explicitly retain their selected direct actuator command instead. The mask
    exposes this fallback (position-only tasks do not call this QP path).
    Commands are in the existing scaled tau_ff units, not raw actor units.
    """
    zero = torch.zeros_like(selected)
    pd = requested_torque_components(desired_position, zero, position, velocity, parameters)[1]
    gain = parameters['control_motor_strength'].detach() * parameters['control_feedforward_weight'].detach()
    available = gain != 0
    physical = selected - pd
    command = torch.where(available, physical / torch.where(available, gain, torch.ones_like(gain)), zero)
    reconstructed = requested_torque_components(desired_position, command, position, velocity, parameters)[0]
    return torch.where(available, reconstructed, selected), command, physical, available


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
