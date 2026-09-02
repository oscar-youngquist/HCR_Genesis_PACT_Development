"""Dependency-neutral home of the immutable HardPACT ablation matrix."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class HardPACTAblationFeatures:
    variant_id: str
    inverse_loss: bool
    rollout_loss: bool
    execution_qp: bool
    qp_gradient: bool
    projection_objective: str  # "off" | "loss" | "metric"
    soft_constraint_penalty: bool

    @property
    def differentiable_qp(self):
        return self.execution_qp and self.qp_gradient

    @property
    def projection_loss(self):
        return self.projection_objective == "loss"

    @property
    def projection_metric(self):
        return self.projection_objective != "off"

    @property
    def needs_bard(self):
        return self.inverse_loss or self.rollout_loss or self.execution_qp

    def scalar_flags(self):
        return {name: float(getattr(self, name)) for name in (
            "inverse_loss", "rollout_loss", "execution_qp", "qp_gradient",
            "projection_loss", "projection_metric", "soft_constraint_penalty",
        )}


HARD_PACT_ABLATIONS: Mapping[str, HardPACTAblationFeatures] = MappingProxyType({
    "baseline": HardPACTAblationFeatures("baseline", False, False, False, False, "off", False),
    "soft": HardPACTAblationFeatures("soft", True, True, False, False, "off", False),
    "hard": HardPACTAblationFeatures("hard", False, False, True, True, "loss", False),
    "full": HardPACTAblationFeatures("full", True, True, True, True, "loss", False),
    "stopgrad": HardPACTAblationFeatures("stopgrad", True, True, True, False, "metric", False),
    "soft_penalty": HardPACTAblationFeatures("soft_penalty", False, False, False, False, "off", True),
    "inverse": HardPACTAblationFeatures("inverse", True, False, False, False, "off", False),
    "rollout": HardPACTAblationFeatures("rollout", False, True, False, False, "off", False),
})
HARD_PACT_VARIANTS = tuple(HARD_PACT_ABLATIONS)
HARD_PACT_BACKENDS = ("genesis", "isaacgym", "isaaclab")


def resolve_hard_pact_features(value="full"):
    if isinstance(value, HardPACTAblationFeatures):
        value = value.variant_id
    elif isinstance(value, Mapping):
        value = value.get("variant_id", value.get("variant", "full"))
    elif not isinstance(value, str):
        value = getattr(value, "variant_id", getattr(value, "variant", "full"))
    try:
        return HARD_PACT_ABLATIONS[str(value)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown HardPACT ablation {value!r}; expected one of "
            f"{', '.join(HARD_PACT_VARIANTS)}"
        ) from exc


def hard_pact_task_names(include_aliases=True):
    names = [f"go2_hard_pact_{v}_{b}" for v in HARD_PACT_VARIANTS for b in HARD_PACT_BACKENDS]
    if include_aliases:
        names.extend(f"go2_hard_pact_{b}" for b in HARD_PACT_BACKENDS)
    return tuple(names)
