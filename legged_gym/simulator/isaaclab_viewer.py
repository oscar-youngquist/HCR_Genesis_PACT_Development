"""Render-only robot visibility; never deactivate prims or change physics."""
from numbers import Integral
import random


def select_rendered_envs(viewer, num_envs):
    """Sample once, without replacement or changes to training RNG state."""
    count = getattr(viewer, 'rendered_envs_count', None)
    if count is None:
        return getattr(viewer, 'rendered_envs_idx', None)
    if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
        raise ValueError('viewer.rendered_envs_count must be a nonnegative integer or None')
    rng = random.Random(getattr(viewer, 'rendered_envs_seed', 0))
    return sorted(rng.sample(range(num_envs), min(count, num_envs)))


def configure_robot_visibility(stage, num_envs, asset_name, rendered_envs_idx):
    """None shows all; [] hides all. Ignore indices beyond small smoke batches.

    Author an explicit opinion for every clone, including visible clones: they
    may inherit from env_0, which itself can be hidden. Terrain stays visible.
    USD visibility also affects rendered camera images, not contact sensing.
    """
    if rendered_envs_idx is None:
        return
    if any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0
           for i in rendered_envs_idx):
        raise ValueError("viewer.rendered_envs_idx must contain nonnegative integer indices or be None")
    selected = set(rendered_envs_idx)
    from pxr import Sdf, UsdGeom
    with Sdf.ChangeBlock():
        for index in range(num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{index}/{asset_name}")
            if not prim.IsValid():
                raise ValueError(f"Missing Isaac Lab robot prim for environment {index}")
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if index in selected else UsdGeom.Tokens.invisible)
