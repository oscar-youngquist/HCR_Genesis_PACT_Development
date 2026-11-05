"""Applied torso wrench convention shared by B1Z1 physics consumers."""
import torch


def application_lever(simulator):
    """World lever to the actual load point, captured at the initial state.

    The Isaac Lab source supplies a free couple at the link origin. Already
    base-referenced sources must likewise return the base point. A different
    backend must declare its actual convention, never infer it from the CoM.
    """
    source = getattr(simulator, "applied_torso_wrench_source", None)
    if source is None:
        raise RuntimeError("B1Z1 physics replay requires an applied_torso_wrench_source with a verified application point")
    base = simulator.base_pos
    point = source()[2]
    return (point - base).detach().clone()


def about_base(wrench_world, lever_world=None):
    """Shift a free couple exactly once; keep learned force gradients intact.

    Missing legacy replay geometry means origin-applied, never randomized CoM.
    Mass/CoM randomization belongs exclusively to the model inertia/bias.
    """
    if lever_world is None:
        return wrench_world
    return torch.cat((wrench_world[..., :3], wrench_world[..., 3:] +
        torch.cross(lever_world.detach(), wrench_world[..., :3], dim=-1)), -1)
