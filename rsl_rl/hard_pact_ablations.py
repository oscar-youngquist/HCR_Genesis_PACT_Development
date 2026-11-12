"""Dependency-neutral home of the immutable HardPACT ablation matrix."""

from dataclasses import dataclass, replace
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
    action_mode: str = "coupled"
    post_qp: str = "full"  # full | metric | imitation | off
    qp_wrench: bool = True
    fk: bool = True
    ff_support: bool = True

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

# Separate namespace; legacy full still enables representation PINNs.
RSS_HARD_PACT_ABLATIONS = MappingProxyType({
    name: HardPACTAblationFeatures(
        "rss_" + name, False, False, qp, post == "full",
        "off" if not qp else "metric" if post == "metric" else "loss",
        False, action, post, wrench, fk, ff,
    ) for name, action, fk, ff, qp, post, wrench in (
        ("pos_ppo", "position", False, False, False, "off", True),
        ("coupled_ppo", "coupled", False, False, False, "off", True),
        ("guided_no_qp", "coupled", True, True, False, "off", True),
        ("qp_no_post", "coupled", True, True, True, "metric", True),
        ("qp_imitation", "coupled", True, True, True, "imitation", True),
        ("full", "coupled", True, True, True, "full", True),
        ("no_fk", "coupled", False, True, True, "full", True),
        ("no_ff", "coupled", True, False, True, "full", True),
        ("no_channels", "coupled", False, False, True, "full", True),
        ("no_qp_wrench", "coupled", True, True, True, "full", False),
    )
})


def resolve_hard_pact_features(value="full", *, inverse_enabled=True, rollout_enabled=True):
    """Named ablations override switches; None uses configurable full-task losses."""
    configurable = value is None
    if configurable:
        value = "full"
    if isinstance(value, HardPACTAblationFeatures):
        # Already-resolved base-task flags must survive runner -> actor handoff.
        return value
    elif isinstance(value, Mapping):
        value = value.get("variant_id", value.get("variant", "full"))
    elif not isinstance(value, str):
        value = getattr(value, "variant_id", getattr(value, "variant", "full"))
    try:
        features = (RSS_HARD_PACT_ABLATIONS[str(value)[4:]] if str(value).startswith("rss_")
                    else HARD_PACT_ABLATIONS[str(value)])
        if not configurable or (inverse_enabled and rollout_enabled):
            return features
        return replace(features, inverse_loss=features.inverse_loss and bool(inverse_enabled),
                       rollout_loss=features.rollout_loss and bool(rollout_enabled))
    except KeyError as exc:
        raise ValueError(
            f"Unknown HardPACT ablation {value!r}; expected one of "
            f"{', '.join(HARD_PACT_VARIANTS)}"
        ) from exc


def hard_pact_task_names(include_aliases=True, include_rss=False):
    # Keep the historical argument name; unsuffixed backend tasks are now
    # config-driven base tasks, not aliases that force the full ablation.
    names = [f"go2_hard_pact_{v}_{b}" for v in HARD_PACT_VARIANTS for b in HARD_PACT_BACKENDS]
    if include_rss:
        names.extend(f"go2_hard_pact_rss_{v}_isaaclab" for v in RSS_HARD_PACT_ABLATIONS)
    if include_aliases:
        names.extend(f"go2_hard_pact_{b}" for b in HARD_PACT_BACKENDS)
    return tuple(names)
