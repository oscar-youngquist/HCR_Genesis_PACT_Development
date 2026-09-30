"""Immutable feature matrix; the full method retains its existing task name."""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class B1Z1Ablation:
    task_name: str
    action_mode: str
    conditioning_mode: str
    representation_pinn_enabled: bool
    actor_phys_enabled: bool


B1Z1_PACT_ABLATIONS = MappingProxyType({
    4: B1Z1Ablation("b1z1_pact_ab4_coupled_none", "coupled", "none", False, False),
    5: B1Z1Ablation("b1z1_pact_ab5_coupled_concat", "coupled", "concat", False, False),
    6: B1Z1Ablation("b1z1_pact_ab6_position_film", "position", "film", False, False),
    7: B1Z1Ablation("b1z1_pact_ab7_position_full", "position", "film", True, True),
    8: B1Z1Ablation("b1z1_pact_ab8_coupled_film", "coupled", "film", False, False),
    9: B1Z1Ablation("b1z1_pact_ab9_representation", "coupled", "film", True, False),
    10: B1Z1Ablation("b1z1_pact_ab10_actor", "coupled", "film", False, True),
    11: B1Z1Ablation("b1z1_pact", "coupled", "film", True, True),
})
