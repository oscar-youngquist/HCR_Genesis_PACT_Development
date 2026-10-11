"""Terrain progress semantics shared by B1Z1 entropy schedules."""


def terrain_entropy_progress(terrain_cfg, observed_level, threshold):
    """No terrain curriculum means its entropy target is already satisfied."""
    if not getattr(terrain_cfg, "curriculum", False):
        return float(threshold)
    return observed_level
