"""Method-neutral performance-gap schedule extracted from HardPACT (same algebra)."""
import torch
from torch.nn import functional as F


def performance_gap_entropy(performance_metrics, bounds, thresholds, temperature):
    keys = ("lin_vel_tracking", "ang_vel_tracking", "terrain_level")
    gaps = [max(0, threshold - performance_metrics.get(key, 0.0)) / threshold
            if threshold > 0 else 0 for key, threshold in zip(keys, thresholds)]
    gaps = torch.tensor(gaps, dtype=torch.float32)
    weighted_gap = torch.sum(F.softmax(gaps / temperature, dim=0) * gaps).item()
    return bounds[0] + weighted_gap * (bounds[1] - bounds[0])
