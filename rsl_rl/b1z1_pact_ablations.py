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
    recruitment: str = "full"
    channel_physics_enabled: bool = True


B1Z1_PACT_ABLATIONS = MappingProxyType({
    4: B1Z1Ablation("b1z1_pact_ab4_coupled_none", "coupled", "none", False, False),
    5: B1Z1Ablation("b1z1_pact_ab5_coupled_concat", "coupled", "concat", False, False),
    6: B1Z1Ablation("b1z1_pact_ab6_position_film", "position", "film", False, False),
    7: B1Z1Ablation("b1z1_pact_ab7_position_full", "position", "film", True, True),
    8: B1Z1Ablation("b1z1_pact_ab8_coupled_film", "coupled", "film", False, False),
    9: B1Z1Ablation("b1z1_pact_ab9_representation", "coupled", "film", True, False),
    10: B1Z1Ablation("b1z1_pact_ab10_actor", "coupled", "film", False, True),
    11: B1Z1Ablation("b1z1_pact", "coupled", "film", True, True),
    12: B1Z1Ablation("b1z1_pos_baseline", "position", "none", False, False),
    13: B1Z1Ablation("b1z1_pact_ab13_workspace", "coupled", "film", True, True, "workspace"),
    14: B1Z1Ablation("b1z1_pact_ab14_no_ee_recruitment", "coupled", "film", True, True, "none"),
    15: B1Z1Ablation("b1z1_pact_ab15_no_channel_physics", "coupled", "film", True, True, "full", False),
})
