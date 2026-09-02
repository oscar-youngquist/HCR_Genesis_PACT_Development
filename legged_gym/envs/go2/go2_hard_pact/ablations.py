"""Task-local public re-export of the one canonical ablation matrix."""

from rsl_rl.hard_pact_ablations import (  # noqa: F401
    HARD_PACT_ABLATIONS, HARD_PACT_BACKENDS, HARD_PACT_VARIANTS,
    HardPACTAblationFeatures, hard_pact_task_names,
    resolve_hard_pact_features,
)

__all__ = [
    "HARD_PACT_ABLATIONS", "HARD_PACT_BACKENDS", "HARD_PACT_VARIANTS",
    "HardPACTAblationFeatures", "hard_pact_task_names",
    "resolve_hard_pact_features",
]
