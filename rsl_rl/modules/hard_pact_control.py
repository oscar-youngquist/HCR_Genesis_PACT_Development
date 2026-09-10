"""Simulator-neutral HardPACT PD/feedforward conversion, in physical Nm.

Callers choose raw or execution-clipped actions before forming q_des/tau_ff.
Actuator effects occur here exactly once. Neither current backend rate-limits
its non-QP controller; QP/held execution retain their separate hard rate box.
"""
import torch


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
