"""Differentiable dynamics adapters used by HardPACT."""

from .bard_go2_dynamics import (
    BARD_FOOT_ORDER,
    BARD_JOINT_ORDER,
    SIMULATOR_FOOT_ORDER,
    SIMULATOR_JOINT_ORDER,
    BardGo2Dynamics,
    Go2BardContext,
    Go2DynamicsTerms,
    build_linear_first_spatial_inertia,
    fixed_mechanics_forward_dynamics,
    simulator_state_to_bard,
    wrench_at_point,
)
from .pinocchio_go2_dynamics import PinocchioGo2Dynamics


def create_go2_dynamics(backend, *args, **kwargs):
    """Construct a Go2 dynamics adapter without changing its consumer API."""
    implementations = {
        "bard": BardGo2Dynamics,
        "pinocchio": PinocchioGo2Dynamics,
    }
    try:
        implementation = implementations[str(backend).lower()]
    except KeyError as error:
        raise ValueError("dynamics backend must be 'bard' or 'pinocchio'") from error
    return implementation(*args, **kwargs)

__all__ = [
    "BARD_FOOT_ORDER",
    "BARD_JOINT_ORDER",
    "SIMULATOR_FOOT_ORDER",
    "SIMULATOR_JOINT_ORDER",
    "BardGo2Dynamics",
    "Go2BardContext",
    "Go2DynamicsTerms",
    "build_linear_first_spatial_inertia",
    "fixed_mechanics_forward_dynamics",
    "simulator_state_to_bard",
    "wrench_at_point",
    "PinocchioGo2Dynamics",
    "create_go2_dynamics",
]
